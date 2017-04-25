#include "config.h"

#include <nmtkit/hybrid_predictor.h>

#include <cmath>
#include <dynet/tensor.h>
#include <nmtkit/array.h>
#include <nmtkit/exception.h>

using std::shared_ptr;
using std::string;
using std::vector;

namespace DE = dynet::expr;

namespace nmtkit {

HybridPredictor::HybridPredictor(
    const unsigned input_size,
    const unsigned softmax_size,
    shared_ptr<BinaryCode> & bc,
    shared_ptr<ErrorCorrectingCode> & ecc,
    const string & loss_type,
    const float softmax_weight,
    const float binary_weight,
    dynet::Model * model)
: softmax_size_(softmax_size)
, num_original_bits_(bc->getNumBits())
, num_encoded_bits_(ecc->getNumBits(bc->getNumBits()))
, bc_(bc)
, ecc_(ecc)
, loss_type_(loss_type)
, softmax_weight_(softmax_weight)
, binary_weight_(binary_weight)
, converter_(
    {input_size, softmax_size + ecc->getNumBits(bc->getNumBits())}, model) {}

void HybridPredictor::prepare(
    dynet::ComputationGraph * cg,
    const bool /* is_training */) {
  converter_.prepare(cg);
}

DE::Expression HybridPredictor::computeLoss(
    const DE::Expression & input,
    const vector<unsigned> & target_ids,
    dynet::ComputationGraph * cg,
    const bool /* is_training */) {
  const unsigned batch_size = target_ids.size();

  // Calculates inner variables.
  vector<unsigned> inner_ids(batch_size);
  vector<float> word_weights_vals(batch_size);
  for (unsigned i = 0; i < batch_size; ++i) {
    inner_ids[i] = target_ids[i] < softmax_size_ ? target_ids[i] : 0;
    word_weights_vals[i] = static_cast<float>(inner_ids[i] == 0);
  }

  // Retrieves target bits.
  vector<float> target_bits(batch_size * num_encoded_bits_);
  for (unsigned b = 0; b < batch_size; ++b) {
    const vector<bool> code = ecc_->encode(bc_->getCode(target_ids[b]));
    const unsigned offset = b * num_encoded_bits_;
    for (unsigned n = 0; n < num_encoded_bits_; ++n) {
      target_bits[offset + n] = static_cast<float>(code[n]);
    }
  }

  // Calculates losses.
  const DE::Expression both_logits = converter_.compute(input);

  const DE::Expression softmax_logits = DE::pickrange(
      both_logits, 0, softmax_size_);
  const DE::Expression softmax_loss = DE::pickneglogsoftmax(
      softmax_logits, inner_ids);

  const DE::Expression binary_logits = DE::pickrange(
          both_logits, softmax_size_, softmax_size_ + num_encoded_bits_);
  const DE::Expression binary_probs = DE::logistic(binary_logits);
  const DE::Expression target_probs = DE::input(
      *cg, dynet::Dim({num_encoded_bits_}, batch_size), target_bits);

  DE::Expression binary_loss;
  if (loss_type_ == "squared") {
    binary_loss = DE::squared_distance(binary_probs, target_probs);
  } else if (loss_type_ == "xent") {
    binary_loss = DE::binary_log_loss(binary_probs, target_probs);
  } else {
    NMTKIT_FATAL("unknown loss type: " + loss_type_);
  }

  const DE::Expression word_weights = DE::input(
      *cg, dynet::Dim({1}, batch_size), word_weights_vals);
  return
    softmax_weight_ * softmax_loss +
    binary_weight_ * word_weights * binary_loss;
}

vector<Predictor::Result> HybridPredictor::predictKBest(
    const DE::Expression & input,
    const unsigned num_results,
    dynet::ComputationGraph * cg) {
  const DE::Expression both_lg = converter_.compute(input);

  const DE::Expression softmax_lg = DE::pickrange(both_lg, 0, softmax_size_);
  const DE::Expression softmax_lp = DE::log_softmax(softmax_lg);
  const vector<float> softmax_lp_val = dynet::as_vector(
      cg->incremental_forward(softmax_lp));
  const unsigned softmax_id = Array::argmax(softmax_lp_val);

  if (softmax_id > 0) {
    return {{softmax_id, softmax_lp_val[softmax_id]}};
  }

  const DE::Expression binary_lg = DE::pickrange(
      both_lg, softmax_size_, softmax_size_ + num_encoded_bits_);
  const DE::Expression binary_p = DE::logistic(binary_lg);
  const vector<float> binary_p_val = dynet::as_vector(
      cg->incremental_forward(binary_p));
  const vector<float> decoded = ecc_->decode(binary_p_val);

  vector<bool> best_code(num_original_bits_);
  float best_log_prob = softmax_lp_val[0];
  for (unsigned i = 0; i < num_original_bits_; ++i) {
    const float x = decoded[i];
    if (x >= 0.5f) {
      best_code[i] = true;
      best_log_prob += log(x);
    } else {
      best_code[i] = false;
      best_log_prob += log(1.0f - x);
    }
  }

  const unsigned wid = bc_->getID(best_code);
  return {{wid != BinaryCode::INVALID_CODE ? wid : 0, best_log_prob}};
}

vector<Predictor::Result> HybridPredictor::predictByIDs(
    const DE::Expression & input,
    const vector<unsigned> word_ids,
    dynet::ComputationGraph * cg) {
  const DE::Expression both_lg = converter_.compute(input);

  const DE::Expression softmax_lg = DE::pickrange(both_lg, 0, softmax_size_);
  const DE::Expression softmax_lp = DE::log_softmax(softmax_lg);
  const vector<float> softmax_lp_val = dynet::as_vector(
      cg->incremental_forward(softmax_lp));

  const DE::Expression binary_lg = DE::pickrange(
      both_lg, softmax_size_, softmax_size_ + num_encoded_bits_);
  const DE::Expression binary_p = DE::logistic(binary_lg);
  const vector<float> binary_p_val = dynet::as_vector(
      cg->incremental_forward(binary_p));
  const vector<float> decoded = ecc_->decode(binary_p_val);

  vector<vector<float>> log_probs(num_original_bits_, {-1e10f, -1e10f});
  for (unsigned i = 0; i < num_original_bits_; ++i) {
    const float x = decoded[i];
    if (x >= 0.5f) {
      log_probs[i][1] = log(x);
      if (x < 1.0f) {
        log_probs[i][0] = log(1.0f - x);
      }
    } else {
      log_probs[i][0] = log(1.0f - x);
      if (x > 0.0f) {
        log_probs[i][1] = log(x);
      }
    }
  }

  vector<Predictor::Result> results;
  for (const unsigned word_id : word_ids) {
    if (word_id > 0 && word_id < softmax_size_) {
      results.emplace_back(
          Predictor::Result {word_id, softmax_lp_val[word_id]});
    } else {
      float log_prob = softmax_lp_val[0];
      const vector<bool> code = bc_->getCode(word_id);
      for (unsigned i = 0; i < num_original_bits_; ++i) {
        log_prob += log_probs[i][code[i]];
      }
      results.emplace_back(Predictor::Result {word_id, log_prob});
    }
  }

  return results;
}

}  // namespace nmtkit

NMTKIT_SERIALIZATION_IMPL(nmtkit::HybridPredictor);

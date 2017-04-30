#include "config.h"

#include <nmtkit/binary_code_predictor.h>

#include <cmath>
#include <dynet/tensor.h>
#include <nmtkit/exception.h>

using std::shared_ptr;
using std::string;
using std::vector;

namespace DE = dynet::expr;

namespace nmtkit {

BinaryCodePredictor::BinaryCodePredictor(
    const unsigned input_size,
    shared_ptr<BinaryCode> & bc,
    shared_ptr<ErrorCorrectingCode> & ecc,
    const string & loss_type,
    dynet::Model * model)
: num_original_bits_(bc->getNumBits())
, num_encoded_bits_(ecc->getNumBits(bc->getNumBits()))
, bc_(bc)
, ecc_(ecc)
, loss_type_(loss_type)
, converter_({input_size, ecc->getNumBits(bc->getNumBits())}, model) {}

void BinaryCodePredictor::prepare(
    dynet::ComputationGraph * cg,
    const bool /* is_training */) {
  converter_.prepare(cg);
}

DE::Expression BinaryCodePredictor::computeLoss(
    const DE::Expression & input,
    const vector<unsigned> & target_ids,
    dynet::ComputationGraph * cg,
    const bool /* is_training */) {
  const unsigned batch_size = target_ids.size();

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
  const DE::Expression output_expr = DE::logistic(converter_.compute(input));
  const DE::Expression target_expr = DE::input(
      *cg, dynet::Dim({num_encoded_bits_}, batch_size), target_bits);
  if (loss_type_ == "squared") {
    return DE::squared_distance(output_expr, target_expr);
  } else if (loss_type_ == "xent") {
    return DE::binary_log_loss(output_expr, target_expr);
  } else {
    NMTKIT_FATAL("unknown loss type: " + loss_type_);
  }
}

vector<Predictor::Result> BinaryCodePredictor::predictKBest(
    const DE::Expression & input,
    const unsigned num_results,
    dynet::ComputationGraph * cg) {
  const DE::Expression output_expr = DE::logistic(converter_.compute(input));
  const vector<float> output = dynet::as_vector(
      cg->incremental_forward(output_expr));
  const vector<float> decoded = ecc_->decode(output);

  vector<bool> best_code(num_original_bits_);
  float best_log_prob = 0.0f;
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

vector<Predictor::Result> BinaryCodePredictor::predictByIDs(
    const DE::Expression & input,
    const vector<unsigned> word_ids,
    dynet::ComputationGraph * cg) {
  const DE::Expression output_expr = DE::logistic(converter_.compute(input));
  const vector<float> output = dynet::as_vector(
      cg->incremental_forward(output_expr));
  const vector<float> decoded = ecc_->decode(output);

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
    float log_prob = 0.0f;
    const vector<bool> code = bc_->getCode(word_id);
    for (unsigned i = 0; i < num_original_bits_; ++i) {
      log_prob += log_probs[i][code[i]];
    }
    results.emplace_back(Predictor::Result {word_id, log_prob});
  }

  return results;
}

Predictor::Result BinaryCodePredictor::sample(
    const DE::Expression & /* input */,
    dynet::ComputationGraph * /* cg */) {
  NMTKIT_FATAL("Not implemented.");
}

}  // namespace nmtkit

NMTKIT_SERIALIZATION_IMPL(nmtkit::BinaryCodePredictor);

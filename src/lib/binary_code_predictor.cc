#include "config.h"

#include <nmtkit/binary_code_predictor.h>

#include <cmath>
#include <dynet/tensor.h>
#include <nmtkit/exception.h>

using namespace std;

namespace DE = dynet::expr;

namespace nmtkit {

BinaryCodePredictor::BinaryCodePredictor(
    const unsigned input_size,
    boost::shared_ptr<BinaryCode> & bc,
    dynet::Model * model)
: bc_(bc)
, converter_({input_size, bc->getNumBits()}, model) {}

void BinaryCodePredictor::prepare(dynet::ComputationGraph * cg) {
  converter_.prepare(cg);
}

DE::Expression BinaryCodePredictor::computeLoss(
    const DE::Expression & input,
    const vector<unsigned> & target_ids,
    dynet::ComputationGraph * cg) {
  const unsigned batch_size = target_ids.size();
  const unsigned num_bits = bc_->getNumBits();

  // Retrieves target bits.
  vector<float> target_bits(batch_size * num_bits);
  for (unsigned b = 0; b < batch_size; ++b) {
    const vector<bool> code = bc_->getCode(target_ids[b]);
    const unsigned offset = b * num_bits;
    for (unsigned n = 0; n < num_bits; ++n) {
      target_bits[offset + n] = static_cast<float>(code[n]);
    }
  }

  // Calculates losses.
  const DE::Expression output_expr = DE::logistic(converter_.compute(input));
  const DE::Expression target_expr = DE::input(*cg, {num_bits}, target_bits);
  return DE::squared_distance(output_expr, target_expr);
}

vector<Predictor::Result> BinaryCodePredictor::predictKBest(
    const DE::Expression & input,
    const unsigned num_results,
    dynet::ComputationGraph * cg) {
  const unsigned num_bits = bc_->getNumBits();

  const vector<dynet::real> logits = dynet::as_vector(
      cg->incremental_forward(converter_.compute(input)));
  vector<bool> best_code(num_bits);
  float best_log_prob = 0.0f;
  //vector<vector<float>> log_probs(num_bits, {0, 0});
  for (unsigned i = 0; i < num_bits; ++i) {
    const float x = static_cast<float>(logits[i]);
    if (x >= 0.0f) {
      best_code[i] = true;
      //log_probs[i][1] = -log(1.0f + exp(-x));
      //log_probs[i][0] = log_probs[i][1] - x;
      best_log_prob += -log(1.0f + exp(-x));
    } else {
      best_code[i] = false;
      //log_probs[i][0] = -log(1.0f + exp(x));
      //log_probs[i][1] = log_probs[i][0] + x;
      best_log_prob += -log(1.0f + exp(x));
    }
  }

  const unsigned wid = bc_->getID(best_code);
  return {{wid != BinaryCode::INVALID_CODE ? wid : 0, best_log_prob}};
}

vector<Predictor::Result> BinaryCodePredictor::predictByIDs(
    const DE::Expression & input,
    const vector<unsigned> word_ids,
    dynet::ComputationGraph * cg) {
  const unsigned num_bits = bc_->getNumBits();

  const vector<dynet::real> logits = dynet::as_vector(
      cg->incremental_forward(converter_.compute(input)));
  vector<vector<float>> log_probs(num_bits, {0.0f, 0.0f});
  for (unsigned i = 0; i < num_bits; ++i) {
    const float x = static_cast<float>(logits[i]);
    if (x >= 0.0f) {
      log_probs[i][1] = -log(1.0f + exp(-x));
      log_probs[i][0] = log_probs[i][1] - x;
    } else {
      log_probs[i][0] = -log(1.0f + exp(x));
      log_probs[i][1] = log_probs[i][0] + x;
    }
  }

  vector<Predictor::Result> results;
  for (const unsigned word_id : word_ids) {
    float log_prob = 0.0f;
    const vector<bool> code = bc_->getCode(word_id);
    for (unsigned i = 0; i < num_bits; ++i) {
      log_prob += log_probs[i][code[i]];
    }
    results.emplace_back(Predictor::Result {word_id, log_prob});
  }

  return results;
}

}  // namespace nmtkit

NMTKIT_SERIALIZATION_IMPL(nmtkit::BinaryCodePredictor);

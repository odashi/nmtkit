#include "config.h"

#include <nmtkit/binary_code_predictor.h>

#include <cmath>
#include <dynet/tensor.h>
#include <nmtkit/exception.h>

using namespace std;

namespace DE = dynet::expr;

namespace nmtkit {

BinaryCodePredictor::BinaryCodePredictor(boost::shared_ptr<BinaryCode> & bc)
: bc_(bc) {}

DE::Expression BinaryCodePredictor::computeLoss(
    const vector<vector<unsigned>> & target_ids,
    const vector<DE::Expression> & scores,
    dynet::ComputationGraph * cg) {
  NMTKIT_CHECK_EQ(
      target_ids.size(), scores.size() + 1,
      "Mismatches lengths of `target_ids` and `scores`");

  const unsigned batch_size = target_ids[0].size();
  const unsigned target_len = scores.size();
  const unsigned num_bits = bc_->getNumBits();

  vector<DE::Expression> losses;

  for (unsigned i = 0; i < target_len; ++i) {
    // Retrieves target bits.
    vector<float> target_bits(batch_size * num_bits);
    for (unsigned j = 0; j < batch_size; ++j) {
      const vector<bool> code = bc_->getCode(target_ids[i + 1][j]);
      const unsigned offset = j * num_bits;
      for (unsigned k = 0; k < num_bits; ++k) {
        target_bits[offset + k] = static_cast<float>(code[k]);
      }
    }

    // Calculates losses.
    const DE::Expression output_expr = DE::logistic(scores[i]);
    const DE::Expression target_expr = DE::input(*cg, {num_bits}, target_bits);
    losses.emplace_back(DE::squared_distance(output_expr, target_expr));
  }

  // Integrates all losses.
  return DE::sum_batches(DE::sum(losses));
}

vector<Predictor::Result> BinaryCodePredictor::predictKBest(
    const DE::Expression & score,
    unsigned num_results,
    dynet::ComputationGraph * cg) {
  const unsigned num_bits = bc_->getNumBits();

  const vector<dynet::real> logits = dynet::as_vector(
      cg->incremental_forward(score));
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
    const DE::Expression & score,
    const vector<unsigned> word_ids,
    dynet::ComputationGraph * cg) {
  const unsigned num_bits = bc_->getNumBits();

  const vector<dynet::real> logits = dynet::as_vector(
      cg->incremental_forward(score));
  vector<vector<float>> log_probs(num_bits, {0, 0});
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

unsigned BinaryCodePredictor::getScoreSize() const {
  return bc_->getNumBits();
}

}  // namespace nmtkit

NMTKIT_SERIALIZATION_IMPL(nmtkit::BinaryCodePredictor);

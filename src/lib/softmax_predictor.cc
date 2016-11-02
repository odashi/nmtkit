#include "config.h"

#include <nmtkit/softmax_predictor.h>

#include <dynet/tensor.h>
#include <nmtkit/array.h>
#include <nmtkit/exception.h>

using namespace std;

namespace DE = dynet::expr;

namespace nmtkit {

SoftmaxPredictor::SoftmaxPredictor(unsigned vocab_size)
: vocab_size_(vocab_size) {
}

DE::Expression SoftmaxPredictor::computeLoss(
    const vector<vector<unsigned>> & target_ids,
    const vector<DE::Expression> & logits) {
  NMTKIT_CHECK_EQ(
      target_ids.size(), logits.size() + 1, "Invalid input lengths.");

  const unsigned tl = logits.size();
  vector<DE::Expression> losses;
  for (unsigned i = 0; i < tl; ++i) {
    losses.emplace_back(
        DE::pickneglogsoftmax(logits[i], target_ids[i + 1]));
  }
  return DE::sum_batches(DE::sum(losses));
}

vector<PredictorResult> SoftmaxPredictor::predictKBest(
    const DE::Expression & logit,
    unsigned num_results,
    dynet::ComputationGraph * cg) {
  NMTKIT_CHECK(
      num_results <= vocab_size_,
      "num_results should not be less than or equal to the vocabulary size.");

  DE::Expression log_probs_expr = DE::log_softmax(logit);
  vector<dynet::real> log_probs = dynet::as_vector(
      cg->incremental_forward(log_probs_expr));

  vector<unsigned> kbest_ids;
  Array::kbest(log_probs, num_results, &kbest_ids);

  vector<PredictorResult> results;
  for (const unsigned word_id : kbest_ids) {
    float log_prob = static_cast<float>(log_probs[word_id]);
    results.emplace_back(PredictorResult {word_id, log_prob});
  }

  return results;
}

vector<PredictorResult> SoftmaxPredictor::predictByIDs(
    const DE::Expression & logit,
    const vector<unsigned> word_ids,
    dynet::ComputationGraph * cg) {
  DE::Expression log_probs_expr = DE::log_softmax(logit);
  vector<dynet::real> log_probs = dynet::as_vector(
      cg->incremental_forward(log_probs_expr));

  vector<PredictorResult> results;
  for (const unsigned word_id : word_ids) {
    float log_prob = static_cast<float>(log_probs[word_id]);
    results.emplace_back(PredictorResult {word_id, log_prob});
  }

  return results;
}

}  // namespace nmtkit

NMTKIT_SERIALIZATION_IMPL(nmtkit::SoftmaxPredictor);

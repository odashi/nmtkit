#include <config.h>

#include <nmtkit/softmax_predictor.h>

#include <dynet/tensor.h>
#include <nmtkit/array.h>
#include <nmtkit/exception.h>

using std::vector;

namespace DE = dynet::expr;

namespace nmtkit {

SoftmaxPredictor::SoftmaxPredictor(
    const unsigned input_size,
    const unsigned vocab_size,
    dynet::Model * model)
: vocab_size_(vocab_size)
, converter_({input_size, vocab_size}, model) {
}

void SoftmaxPredictor::prepare(
    dynet::ComputationGraph * cg,
    const bool /* is_training */) {
  converter_.prepare(cg);
}

DE::Expression SoftmaxPredictor::computeLoss(
    const DE::Expression & input,
    const vector<unsigned> & target_ids,
    dynet::ComputationGraph * /*cg*/,
    const bool /* is_training */) {
  const DE::Expression score = converter_.compute(input);
  return DE::pickneglogsoftmax(score, target_ids);
}

vector<Predictor::Result> SoftmaxPredictor::predictKBest(
    const DE::Expression & input,
    const unsigned num_results,
    dynet::ComputationGraph * cg) {
  NMTKIT_CHECK(
      num_results <= vocab_size_,
      "num_results should not be less than or equal to the vocabulary size.");

  const DE::Expression score = converter_.compute(input);
  const DE::Expression log_probs_expr = DE::log_softmax(score);
  vector<float> log_probs = dynet::as_vector(
      cg->incremental_forward(log_probs_expr));
  NMTKIT_CHECK(
      log_probs.size() == vocab_size_,
      "Size of resulting log-prob array is incorrect. "
      "Attempting to decode multiple sentences?");

  vector<Predictor::Result> results;
  for (const unsigned word_id : Array::kbest(log_probs, num_results)) {
    results.emplace_back(Predictor::Result {word_id, log_probs[word_id]});
  }

  return results;
}

vector<Predictor::Result> SoftmaxPredictor::predictByIDs(
    const DE::Expression & input,
    const vector<unsigned> word_ids,
    dynet::ComputationGraph * cg) {
  const DE::Expression score = converter_.compute(input);
  const DE::Expression log_probs_expr = DE::log_softmax(score);
  vector<float> log_probs = dynet::as_vector(
      cg->incremental_forward(log_probs_expr));
  NMTKIT_CHECK(
      log_probs.size() == vocab_size_,
      "Size of resulting log-prob array is incorrect. "
      "Attempting to decode multiple sentences?");

  vector<Predictor::Result> results;
  for (const unsigned word_id : word_ids) {
    NMTKIT_CHECK(
        word_id < vocab_size_,
        "Each word_id should be less than vocab_size.");
    results.emplace_back(Predictor::Result {word_id, log_probs[word_id]});
  }

  return results;
}

Predictor::Result SoftmaxPredictor::sample(
    const DE::Expression & input,
    dynet::ComputationGraph * cg) {
  const DE::Expression score = converter_.compute(input);

  // log(P)
  const DE::Expression log_probs_expr = DE::log_softmax(score);
  vector<float> log_probs = dynet::as_vector(
      cg->incremental_forward(log_probs_expr));
  NMTKIT_CHECK(
      log_probs.size() == vocab_size_,
      "Size of resulting log-prob array is incorrect. "
      "Attempting to decode multiple sentences?");

  // Gumbel-softmax sampling
  const DE::Expression noise = DE::random_gumbel(
      *cg, log_probs_expr.dim(), 0.0f, 1.0f);
  vector<float> id_scores = dynet::as_vector(
      cg->incremental_forward(log_probs_expr + noise));
  const unsigned argmax_id = Array::argmax(id_scores);

  return Predictor::Result { argmax_id, log_probs[argmax_id] };
}

}  // namespace nmtkit

NMTKIT_SERIALIZATION_IMPL(nmtkit::SoftmaxPredictor);

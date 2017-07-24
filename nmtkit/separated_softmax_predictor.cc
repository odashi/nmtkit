#include "config.h"

#include <nmtkit/separated_softmax_predictor.h>

#include <cmath>
#include <dynet/tensor.h>
#include <nmtkit/array.h>
#include <nmtkit/exception.h>

using std::shared_ptr;
using std::string;
using std::vector;

namespace DE = dynet::expr;

namespace nmtkit {

SeparatedSoftmaxPredictor::SeparatedSoftmaxPredictor(
    const unsigned input_size,
    const unsigned vocab_size,
    const unsigned first_size,
    dynet::Model * model)
: vocab_size_(vocab_size)
, first_size_(first_size)
, first_converter_({input_size, first_size}, model)
, second_converter_({input_size, vocab_size}, model) {}

void SeparatedSoftmaxPredictor::prepare(
    dynet::ComputationGraph * cg,
    const bool /* is_training */) {
  first_converter_.prepare(cg);
  second_converter_.prepare(cg);
}

DE::Expression SeparatedSoftmaxPredictor::computeLoss(
    const DE::Expression & input,
    const vector<unsigned> & target_ids,
    dynet::ComputationGraph * cg,
    const bool /* is_training */) {
  const unsigned batch_size = target_ids.size();

  // Calculates inner variables.
  vector<unsigned> inner_ids(batch_size);
  vector<float> word_weights_vals(batch_size);
  for (unsigned i = 0; i < batch_size; ++i) {
    inner_ids[i] = target_ids[i] < first_size_ ? target_ids[i] : 0;
    word_weights_vals[i] = static_cast<float>(inner_ids[i] == 0);
  }

  // Calculates losses.
  const DE::Expression first_logits = first_converter_.compute(input);
  const DE::Expression second_logits = second_converter_.compute(input);

  const DE::Expression first_loss = DE::pickneglogsoftmax(
      first_logits, inner_ids);
  const DE::Expression second_loss = DE::pickneglogsoftmax(
      second_logits, target_ids);

  const DE::Expression word_weights = DE::input(
      *cg, dynet::Dim({1}, batch_size), word_weights_vals);
  return first_loss + word_weights * second_loss;
}

vector<Predictor::Result> SeparatedSoftmaxPredictor::predictKBest(
    const DE::Expression & input,
    const unsigned num_results,
    dynet::ComputationGraph * cg) {
  const DE::Expression first_lg = first_converter_.compute(input);
  const DE::Expression first_lp = DE::log_softmax(first_lg);
  const vector<float> first_lp_val = dynet::as_vector(
      cg->incremental_forward(first_lp));
  const unsigned first_id = Array::argmax(first_lp_val);

  if (first_id > 0) {
    return {{first_id, first_lp_val[first_id]}};
  }

  const DE::Expression second_lg = second_converter_.compute(input);
  const DE::Expression second_lp = DE::log_softmax(second_lg);
  const vector<float> second_lp_val = dynet::as_vector(
      cg->incremental_forward(second_lp));
  const unsigned second_id = Array::argmax(second_lp_val);

  return {{second_id, first_lp_val[0] + second_lp_val[second_id]}};
}

vector<Predictor::Result> SeparatedSoftmaxPredictor::predictByIDs(
    const DE::Expression & input,
    const vector<unsigned> word_ids,
    dynet::ComputationGraph * cg) {
  const DE::Expression first_lg = first_converter_.compute(input);
  const DE::Expression second_lg = second_converter_.compute(input);
  const DE::Expression first_lp = DE::log_softmax(first_lg);
  const DE::Expression second_lp = DE::log_softmax(second_lg);
  const vector<float> first_lp_val = dynet::as_vector(
      cg->incremental_forward(first_lp));
  const vector<float> second_lp_val = dynet::as_vector(
      cg->incremental_forward(second_lp));

  vector<Predictor::Result> results;
  for (const unsigned word_id : word_ids) {
    if (word_id > 0 && word_id < first_size_) {
      results.emplace_back(Predictor::Result {word_id, first_lp_val[word_id]});
    } else {
      results.emplace_back(Predictor::Result {
          word_id, first_lp_val[0] + second_lp_val[word_id]});
    }
  }

  return results;
}

Predictor::Result SeparatedSoftmaxPredictor::sample(
    const DE::Expression & /* input */,
    dynet::ComputationGraph * /* cg */) {
  NMTKIT_FATAL("Not implemented.");
}

}  // namespace nmtkit

NMTKIT_SERIALIZATION_IMPL(nmtkit::SeparatedSoftmaxPredictor);

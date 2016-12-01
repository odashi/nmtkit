#include "config.h"

#include <nmtkit/backward_encoder.h>

#include <nmtkit/array.h>

using namespace std;

namespace DE = dynet::expr;

namespace nmtkit {

BackwardEncoder::BackwardEncoder(
    unsigned num_layers,
    unsigned vocab_size,
    unsigned embed_size,
    unsigned hidden_size,
    dynet::Model * model)
: num_layers_(num_layers)
, vocab_size_(vocab_size)
, embed_size_(embed_size)
, hidden_size_(hidden_size)
, rnn_(num_layers, embed_size, hidden_size, model) {
  p_lookup_ = model->add_lookup_parameters(vocab_size_, {embed_size_});
}

void BackwardEncoder::prepare(
    const float dropout_ratio,
    dynet::ComputationGraph * cg) {
  rnn_.set_dropout(dropout_ratio);
  rnn_.new_graph(*cg);
  rnn_.start_new_sequence();
}

vector<DE::Expression> BackwardEncoder::compute(
    const vector<vector<unsigned>> & input_ids,
    dynet::ComputationGraph * cg) {
  vector<DE::Expression> outputs;
  for (auto it = input_ids.rbegin(); it != input_ids.rend(); ++it) {
    const DE::Expression embed = DE::lookup(*cg, p_lookup_, *it);
    outputs.emplace_back(rnn_.add_input(embed));
  }
  Array::reverse(&outputs);
  return outputs;
}

vector<DE::Expression> BackwardEncoder::getStates() const {
  return rnn_.final_s();
}

}  // namespace nmtkit

NMTKIT_SERIALIZATION_IMPL(nmtkit::BackwardEncoder);

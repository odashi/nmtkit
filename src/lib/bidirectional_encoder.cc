#include "config.h"

#include <nmtkit/bidirectional_encoder.h>

#include <nmtkit/array.h>

using namespace std;

namespace DE = dynet::expr;

namespace nmtkit {

BidirectionalEncoder::BidirectionalEncoder(
    unsigned num_layers,
    unsigned vocab_size,
    unsigned embed_size,
    unsigned hidden_size,
    dynet::Model * model)
: num_layers_(num_layers)
, vocab_size_(vocab_size)
, embed_size_(embed_size)
, hidden_size_(hidden_size)
, rnn_fw_(num_layers, embed_size, hidden_size, model)
, rnn_bw_(num_layers, embed_size, hidden_size, model) {
  p_lookup_ = model->add_lookup_parameters(vocab_size_, {embed_size_});
}

void BidirectionalEncoder::prepare(
    const float dropout_ratio,
    dynet::ComputationGraph * cg) {
  rnn_fw_.set_dropout(dropout_ratio);
  rnn_bw_.set_dropout(dropout_ratio);
  rnn_fw_.new_graph(*cg);
  rnn_bw_.new_graph(*cg);
  rnn_fw_.start_new_sequence();
  rnn_bw_.start_new_sequence();
}

vector<DE::Expression> BidirectionalEncoder::compute(
    const vector<vector<unsigned>> & input_ids,
    dynet::ComputationGraph * cg) {
  const int input_len = input_ids.size();

  // Embedding lookup
  vector<DE::Expression> embeds;
  for (int i = 0; i < input_len; ++i) {
    embeds.emplace_back(DE::lookup(*cg, p_lookup_, input_ids[i]));
  }

  // Forward encoding
  vector<DE::Expression> fw_outputs;
  for (int i = 0; i < input_len; ++i) {
    fw_outputs.emplace_back(rnn_fw_.add_input(embeds[i]));
  }

  // Backward encoding
  vector<DE::Expression> bw_outputs;
  for (int i = input_len - 1; i >= 0; --i) {
    bw_outputs.emplace_back(rnn_bw_.add_input(embeds[i]));
  }
  Array::reverse(&bw_outputs);

  // Make outputs.
  vector<DE::Expression> outputs;
  for (int i = 0; i < input_len; ++i) {
    outputs.emplace_back(DE::concatenate({fw_outputs[i], bw_outputs[i]}));
  }
  return outputs;
}

vector<DE::Expression> BidirectionalEncoder::getStates() const {
  // Make new states by concatenating forward/backward states.
  const vector<DE::Expression> fw_states = rnn_fw_.final_s();
  const vector<DE::Expression> bw_states = rnn_bw_.final_s();
  vector<DE::Expression> states;
  for (unsigned i = 0; i < fw_states.size(); ++i) {
    states.emplace_back(DE::concatenate({fw_states[i], bw_states[i]}));
  }
  return states;
}

}  // namespace nmtkit

NMTKIT_SERIALIZATION_IMPL(nmtkit::BidirectionalEncoder);

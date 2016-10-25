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

void BidirectionalEncoder::build(
    const vector<vector<unsigned>> & input_ids,
    dynet::ComputationGraph * cg,
    std::vector<DE::Expression> * output_states,
    DE::Expression * final_state) {
  const int input_len = input_ids.size();

  // Embedding lookup
  vector<DE::Expression> embeds;
  for (int i = 0; i < input_len; ++i) {
    embeds.emplace_back(DE::lookup(*cg, p_lookup_, input_ids[i]));
  }

  // Forward encoding
  rnn_fw_.new_graph(*cg);
  rnn_fw_.start_new_sequence();
  vector<DE::Expression> fw_outputs;
  for (int i = 0; i < input_len; ++i) {
    fw_outputs.emplace_back(rnn_fw_.add_input(embeds[i]));
  }

  // Backward encoding
  rnn_bw_.new_graph(*cg);
  rnn_bw_.start_new_sequence();
  vector<DE::Expression> bw_outputs;
  for (int i = input_len - 1; i >= 0; --i) {
    bw_outputs.emplace_back(rnn_bw_.add_input(embeds[i]));
  }
  Array::reverse(&bw_outputs);

  // Make output states.
  if (output_states != nullptr) {
    output_states->clear();
    for (int i = 0; i < input_len; ++i) {
      output_states->emplace_back(
          DE::concatenate({fw_outputs[i], bw_outputs[i]}));
    }
  }

  // Make the final state.
  if (final_state != nullptr) {
    *final_state = DE::concatenate({fw_outputs.back(), bw_outputs.front()});
  }
}

}  // namespace nmtkit

NMTKIT_SERIALIZATION_IMPL(nmtkit::BidirectionalEncoder);

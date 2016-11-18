#include "config.h"

#include <nmtkit/bahdanau_decoder.h>
#include <nmtkit/exception.h>

using namespace std;

namespace DE = dynet::expr;

namespace nmtkit {

BahdanauDecoder::BahdanauDecoder(
    unsigned vocab_size,
    unsigned in_embed_size,
    unsigned out_embed_size,
    unsigned hidden_size,
    unsigned seed_size,
    unsigned context_size,
    dynet::Model * model)
: vocab_size_(vocab_size)
, in_embed_size_(in_embed_size)
, out_embed_size_(out_embed_size)
, hidden_size_(hidden_size)
, seed_size_(seed_size)
, context_size_(context_size)
// 3-layer conversion between encoder and decoder.
, enc2dec_({seed_size, (seed_size + hidden_size) / 2, hidden_size}, model)
, dec2out_({in_embed_size + context_size + hidden_size, out_embed_size}, model)
, rnn_(1, in_embed_size + context_size, hidden_size, model)
, p_lookup_(model->add_lookup_parameters(vocab_size, {in_embed_size}))
{}

Decoder::State BahdanauDecoder::prepare(
    const DE::Expression & seed,
    dynet::ComputationGraph * cg) {
  // NOTE: LSTMBuilder::start_new_sequence() takes initial states with below
  //       layout:
  //         {c1, c2, ..., cn, h1, h2, ..., hn}
  //       where cx is the initial cell states and hx is the initial outputs.
  enc2dec_.prepare(cg);
  dec2out_.prepare(cg);
  const DE::Expression init_c = enc2dec_.compute(seed, cg);
  const DE::Expression init_h = DE::tanh(init_c);
  rnn_.new_graph(*cg);
  rnn_.start_new_sequence({init_c, init_h});
  return {{rnn_.state()}, {init_h}};
}

Decoder::State BahdanauDecoder::oneStep(
    const Decoder::State & state,
    const vector<unsigned> & input_ids,
    Attention * attention,
    dynet::ComputationGraph * cg,
    dynet::expr::Expression * atten_probs,
    dynet::expr::Expression * output) {
  NMTKIT_CHECK_EQ(
      1, state.positions.size(), "Invalid number of RNN positions.");
  NMTKIT_CHECK_EQ(
      1, state.params.size(), "Invalid number of state parameters.");

  // Aliases
  const dynet::RNNPointer & prev_pos = state.positions[0];
  const DE::Expression & prev_h = state.params[0];

  // Calculation
  const DE::Expression in_embed = DE::lookup(*cg, p_lookup_, input_ids);
  const vector<DE::Expression> atten_info = attention->compute(prev_h, cg);
  const DE::Expression next_h = rnn_.add_input(
      prev_pos, DE::concatenate({in_embed, atten_info[1]}));
  const DE::Expression out_embed = DE::rectify(
      dec2out_.compute(
          DE::concatenate({in_embed, atten_info[1], next_h}), cg));

  // Store outputs.
  if (atten_probs != nullptr) {
    *atten_probs = atten_info[0];
  }
  if (output != nullptr) {
    *output = out_embed;
  }

  return {{rnn_.state()}, {next_h}};
}

}  // namespace nmtkit

NMTKIT_SERIALIZATION_IMPL(nmtkit::BahdanauDecoder);

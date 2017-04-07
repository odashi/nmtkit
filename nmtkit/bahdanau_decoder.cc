#include <nmtkit/bahdanau_decoder.h>
#include <nmtkit/exception.h>

using std::vector;

namespace DE = dynet::expr;

namespace nmtkit {

BahdanauDecoder::BahdanauDecoder(
    unsigned num_layers,
    unsigned vocab_size,
    unsigned in_embed_size,
    unsigned out_embed_size,
    unsigned hidden_size,
    unsigned seed_size,
    unsigned context_size,
    dynet::Model * model)
: num_layers_(num_layers)
, vocab_size_(vocab_size)
, in_embed_size_(in_embed_size)
, out_embed_size_(out_embed_size)
, hidden_size_(hidden_size)
, seed_size_(seed_size)
, context_size_(context_size)
, dec2out_({in_embed_size + context_size + hidden_size, out_embed_size}, model)
, rnn_(num_layers, in_embed_size + context_size, hidden_size, *model)
, p_lookup_(model->add_lookup_parameters(vocab_size, {in_embed_size}))
{
  for (unsigned i = 0; i < num_layers; ++i) {
    enc2dec_.emplace_back(
        MultilayerPerceptron({seed_size, hidden_size}, model));
  }
}

Decoder::State BahdanauDecoder::prepare(
    const vector<DE::Expression> & seed,
    const float dropout_ratio,
    dynet::ComputationGraph * cg) {
  NMTKIT_CHECK_EQ(2 * num_layers_, seed.size(), "Invalid number of initial states.");
  vector<DE::Expression> states;
  for (unsigned i = 0; i < num_layers_; ++i) {
    enc2dec_[i].prepare(cg);
    states.emplace_back(enc2dec_[i].compute(seed[i]));
  }
  for (unsigned i = 0; i < num_layers_; ++i) {
    states.emplace_back(DE::tanh(states[i]));
  }
  rnn_.set_dropout(dropout_ratio);
  rnn_.new_graph(*cg);
  rnn_.start_new_sequence(states);
  dec2out_.prepare(cg);
  return {{rnn_.state()}, {rnn_.back()}};
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
  const vector<DE::Expression> atten_info = attention->compute(prev_h);
  const DE::Expression next_h = rnn_.add_input(
      prev_pos, DE::concatenate({in_embed, atten_info[1]}));
  // Note: In the original implementation, the MaxOut function is used for the
  //       output nonlinearization, not ReLU.
  const DE::Expression out_embed = DE::rectify(
      dec2out_.compute(DE::concatenate({in_embed, atten_info[1], next_h})));

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

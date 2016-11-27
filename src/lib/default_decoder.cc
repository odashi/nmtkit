#include "config.h"

#include <nmtkit/default_decoder.h>
#include <nmtkit/exception.h>

using namespace std;

namespace DE = dynet::expr;

namespace nmtkit {

DefaultDecoder::DefaultDecoder(
    unsigned vocab_size,
    unsigned embed_size,
    unsigned hidden_size,
    unsigned seed_size,
    unsigned context_size,
    dynet::Model * model)
: vocab_size_(vocab_size)
, embed_size_(embed_size)
, hidden_size_(hidden_size)
, seed_size_(seed_size)
, context_size_(context_size)
, rnn_(1, embed_size + context_size, hidden_size, model)
, p_lookup_(model->add_lookup_parameters(vocab_size, {embed_size}))
{
  for (unsigned i = 0; i < 2; ++i) {
    enc2dec_.emplace_back(
        MultilayerPerceptron({seed_size, hidden_size}, model));
  }
}

Decoder::State DefaultDecoder::prepare(
    const vector<DE::Expression> & seed,
    dynet::ComputationGraph * cg) {
  NMTKIT_CHECK_EQ(2, seed.size(), "Invalid number of initial states.");
  vector<DE::Expression> states;
  for (unsigned i = 0; i < 2; ++i) {
    enc2dec_[i].prepare(cg);
    states.emplace_back(enc2dec_[i].compute(seed[i]));
  }
  rnn_.new_graph(*cg);
  rnn_.start_new_sequence(states);
  return {{rnn_.state()}, {rnn_.back()}};
}

Decoder::State DefaultDecoder::oneStep(
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
  const DE::Expression embed = DE::lookup(*cg, p_lookup_, input_ids);
  const vector<DE::Expression> atten_info = attention->compute(prev_h, cg);
  const DE::Expression next_h = rnn_.add_input(
      prev_pos, DE::concatenate({embed, atten_info[1]}));

  // Store outputs.
  if (atten_probs != nullptr) {
    *atten_probs = atten_info[0];
  }
  if (output != nullptr) {
    *output = next_h;
  }

  return {{rnn_.state()}, {next_h}};
}

}  // namespace nmtkit

NMTKIT_SERIALIZATION_IMPL(nmtkit::DefaultDecoder);

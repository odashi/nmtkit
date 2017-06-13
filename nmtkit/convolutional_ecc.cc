#include "config.h"

#include <cmath>
#include <utility>
#include <nmtkit/convolutional_ecc.h>
#include <nmtkit/exception.h>

using std::vector;

namespace nmtkit {

ConvolutionalECC::ConvolutionalECC(const unsigned num_registers)
: num_symbols_(2)
, num_registers_(num_registers) {
  NMTKIT_CHECK(
      num_registers >= 2,
      "num_registers should be greater than or equal to 2.");
  NMTKIT_CHECK(
      num_registers <= 6,
      "num_registers should be less than or equal to 6.");
}

vector<bool> ConvolutionalECC::convolute(
  const vector<bool> & rev_input) const {
  NMTKIT_CHECK(
      rev_input.size() > num_registers_,
      "Length of rev_input is too short.");

  const unsigned num_states = rev_input.size() - num_registers_;
  vector<bool> output(num_symbols_ * num_states);

  for (unsigned i = 0; i < num_states; ++i) {
    const unsigned in_ofs = num_states - i - 1;
    const unsigned out_ofs = num_symbols_ * i;
#define I(x) rev_input[in_ofs + x]
#define O(x) output[out_ofs + x]
    switch (num_registers_) {
      case 2:
        O(0) = I(0) ^ I(2);
        O(1) = I(0) ^ I(1) ^ I(2);
        break;
      case 3:
        O(0) = I(0) ^ I(1) ^ I(3);
        O(1) = I(0) ^ I(1) ^ I(2) ^ I(3);
        break;
      case 4:
        O(0) = I(0) ^ I(3) ^ I(4);
        O(1) = I(0) ^ I(1) ^ I(2) ^ I(4);
        break;
      case 5:
        O(0) = I(0) ^ I(2) ^ I(4) ^ I(5);
        O(1) = I(0) ^ I(1) ^ I(2) ^ I(3) ^ I(5);
        break;
      case 6:
        O(0) = I(0) ^ I(1) ^ I(2) ^ I(3) ^ I(6);
        O(1) = I(0) ^ I(2) ^ I(3) ^ I(5) ^ I(6);
        break;
      default:
        NMTKIT_FATAL("Invalid num_registers_.");
    }
#undef I
#undef O
  }

  return output;
}

vector<bool> ConvolutionalECC::encode(
    const vector<bool> & original_bits) const {
  const unsigned num_original_bits = original_bits.size();
  const unsigned num_output_states = num_original_bits + num_registers_;

  // Constructs input sequence: 0...0 + reversed(original_bits) + 0...0
  vector<bool> input(num_original_bits + 2 * num_registers_, false);
  for (unsigned i = 0; i < num_original_bits; ++i){
    input[num_output_states - i - 1] = original_bits[i];
  }

  return convolute(input);
}

vector<float> ConvolutionalECC::decode(
    const vector<float> & encoded_probs) const {
  NMTKIT_CHECK(
      encoded_probs.size() > 2 * num_registers_,
      "Length of encoded_probs is too short.");
  NMTKIT_CHECK_EQ(
      0, encoded_probs.size() % num_symbols_,
      "Length of encoded_probs should be a multiple of num_symbols_.");

  const unsigned num_nodes = 1 << num_registers_;

  // Structure representing a backward transition in trellis diagrams.
  struct Transition {
    unsigned prev;
    vector<bool> output;
  };

  // Make a trellis diagram.
  vector<vector<Transition>> trans(num_nodes, vector<Transition>(2));

  for (unsigned cur = 0; cur < num_nodes; ++cur) {
    vector<bool> s(num_registers_);
    for (unsigned i = 0; i < num_registers_; ++i) {
      s[i] = static_cast<bool>((cur >> i) & 1);
    }
    for (unsigned last = 0; last < 2; ++last) {
      const unsigned prev = (cur >> 1) | (last << (num_registers_ - 1));
      vector<bool> ss = s;
      ss.push_back(static_cast<bool>(last));
      trans[cur][last] = {prev, convolute(ss)};
      NMTKIT_CHECK(convolute(ss).size() == 2, "check");
    }
  }

  // Decode.
  // This function is now mplementing Viterbi decoding.
  const float NEG_INF = -1e10f;
  vector<float> prev_logprob(num_nodes, NEG_INF);
  prev_logprob[0] = 0.0;
  vector<vector<unsigned>> backptr_list;

  auto mylog = [&](const float x) { return x > 0.0 ? std::log(x) : NEG_INF; };

  for (unsigned pos = 0; pos < encoded_probs.size(); pos += num_symbols_) {
    vector<vector<float>> lp(num_symbols_, vector<float>(2));
    for (unsigned i = 0; i < num_symbols_; ++i) {
      const float prob = encoded_probs[pos + i];
      lp[i][0] = mylog(1.0f - prob);
      lp[i][1] = mylog(prob);
    }
    vector<float> cur_logprob;
    vector<unsigned> cur_backptr;
    for (unsigned cur = 0; cur < num_nodes; ++cur) {
      float best_logprob = NEG_INF;
      unsigned best_backptr = 0;  // dummy
      for (unsigned last = 0; last < 2; ++last) {
        const Transition & tr = trans[cur][last];
        float logprob = prev_logprob[tr.prev];
        for (unsigned i = 0; i < num_symbols_; ++i) {
          logprob += lp[i][tr.output[i]];
        }
        if (logprob > best_logprob) {
          best_logprob = logprob;
          best_backptr = tr.prev;
        }
      }
      cur_logprob.emplace_back(best_logprob);
      cur_backptr.emplace_back(best_backptr);
    }
    prev_logprob = std::move(cur_logprob);
    backptr_list.emplace_back(std::move(cur_backptr));
  }

  // Retrieves Viterbi path.
  vector<unsigned> nodes {0};
  for (auto bp = backptr_list.rbegin(); bp != backptr_list.rend(); ++bp) {
    nodes.emplace_back((*bp)[nodes.back()]);
  }

  // NOTE: Define the log-probability distance between 1 and 0 as 1.0.
  // 1/(1+e), e/(1+e)
  const float P[2] {0.2689414, 0.7310586};

  vector<float> result(nodes.size() - num_registers_ - 1);
  const unsigned result_ofs = nodes.size() - 2;
  for (unsigned i = 0; i < result.size(); ++i) {
    result[i] = P[nodes[result_ofs - i] & 1];
  }

  return result;
}

unsigned ConvolutionalECC::getNumBits(const unsigned num_original_bits) const {
  return num_symbols_ * (num_original_bits + num_registers_);
}

}  // namespace nmtkit

NMTKIT_SERIALIZATION_IMPL(nmtkit::ConvolutionalECC);

#include "config.h"

#include <numeric>
#include <nmtkit/array.h>
#include <nmtkit/frequency_code.h>

using namespace std;

namespace nmtkit {

FrequencyCode::FrequencyCode(const Vocabulary & vocab)
: wid_to_code_(vocab.size())
, code_to_wid_(vocab.size())
, vocab_size_(vocab.size()) {
  // Sort word IDs by word frequency.
  iota(wid_to_code_.begin(), wid_to_code_.end(), 0);
  Array::sort(&wid_to_code_, [&](const unsigned a, const unsigned b) {
      return vocab.getFrequency(a) > vocab.getFrequency(b);
  });

  // Make reverse mapping.
  for (unsigned i = 0; i < vocab_size_; ++i) {
    code_to_wid_[wid_to_code_[i]] = i;
  }

  // Retrieve bit length of code.
  num_bits_ = 0;
  while (1u << num_bits_ < vocab_size_) {
    ++num_bits_;
  }
}

vector<float> FrequencyCode::getCode(const unsigned id) const {
  const unsigned code = wid_to_code_[id];
  vector<float> result(num_bits_);
  for (unsigned i = 0; i < num_bits_; ++i) {
    result[i] = static_cast<float>((code >> i) & 1);
  }
  return result;
}

unsigned FrequencyCode::getID(const vector<float> & probs) const {
  unsigned result = 0;
  for (unsigned i = 0; i < num_bits_; ++i) {
    result |= static_cast<unsigned>(probs[i] >= 0.5f) << i;
  }
  return result < vocab_size_ ? code_to_wid_[result] : 0;
}

unsigned FrequencyCode::getNumBits() const {
  return num_bits_;
}

}  // namespace nmtkit


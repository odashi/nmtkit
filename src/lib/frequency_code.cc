#include "config.h"

#include <numeric>
#include <nmtkit/array.h>
#include <nmtkit/exception.h>
#include <nmtkit/frequency_code.h>

using namespace std;

namespace nmtkit {

FrequencyCode::FrequencyCode(const Vocabulary & vocab)
: wid_to_code_(vocab.size())
, code_to_wid_(vocab.size())
, vocab_size_(vocab.size()) {
  // Sorts word IDs by word frequency.
  iota(wid_to_code_.begin(), wid_to_code_.end(), 0);
  Array::sort(&wid_to_code_, [&](const unsigned a, const unsigned b) {
      return vocab.getFrequency(a) > vocab.getFrequency(b);
  });

  // Makes reverse mapping.
  for (unsigned i = 0; i < vocab_size_; ++i) {
    code_to_wid_[wid_to_code_[i]] = i;
  }

  // Retrieves bit length of code.
  num_bits_ = 0;
  while (1u << num_bits_ < vocab_size_) {
    ++num_bits_;
  }
}

vector<bool> FrequencyCode::getCode(const unsigned id) const {
  NMTKIT_CHECK(id < vocab_size_, "id should be less than vocab_size.");
  const unsigned code = wid_to_code_[id];
  vector<bool> result(num_bits_);
  for (unsigned i = 0; i < num_bits_; ++i) {
    result[i] = !!((code >> i) & 1);
  }
  return result;
}

unsigned FrequencyCode::getID(const vector<bool> & code) const {
  unsigned result = 0;
  for (unsigned i = 0; i < num_bits_; ++i) {
    result |= static_cast<unsigned>(!!code[i]) << i;
  }
  return result < vocab_size_ ? code_to_wid_[result] : BinaryCode::INVALID_CODE;
}

unsigned FrequencyCode::getNumBits() const {
  return num_bits_;
}

}  // namespace nmtkit

NMTKIT_SERIALIZATION_IMPL(nmtkit::FrequencyCode);

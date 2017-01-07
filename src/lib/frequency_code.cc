#include "config.h"

#include <numeric>
#include <nmtkit/array.h>
#include <nmtkit/exception.h>
#include <nmtkit/frequency_code.h>

using namespace std;

namespace nmtkit {

// NOTE(odashi): Currently it uses word ID directly as the bit representation.

FrequencyCode::FrequencyCode(const Vocabulary & vocab)
: vocab_size_(vocab.size()) {
  num_bits_ = 0;
  while (1u << num_bits_ < vocab_size_) {
    ++num_bits_;
  }
}

vector<bool> FrequencyCode::getCode(const unsigned id) const {
  NMTKIT_CHECK(id < vocab_size_, "id should be less than vocab_size.");
  vector<bool> result(num_bits_);
  for (unsigned i = 0; i < num_bits_; ++i) {
    result[i] = static_cast<bool>((id >> i) & 1);
  }
  return result;
}

unsigned FrequencyCode::getID(const vector<bool> & code) const {
  NMTKIT_CHECK_EQ(num_bits_, code.size(), "Invalid length of code.");
  unsigned result = 0;
  for (unsigned i = 0; i < num_bits_; ++i) {
    result |= static_cast<unsigned>(code[i]) << i;
  }
  return result < vocab_size_ ? result : BinaryCode::INVALID_CODE;
}

unsigned FrequencyCode::getNumBits() const {
  return num_bits_;
}

}  // namespace nmtkit

NMTKIT_SERIALIZATION_IMPL(nmtkit::FrequencyCode);

#ifndef NMTKIT_FREQUENCY_CODE_H_
#define NMTKIT_FREQUENCY_CODE_H_

#include <boost/serialization/base_object.hpp>
#include <nmtkit/binary_code.h>
#include <nmtkit/vocabulary.h>

namespace nmtkit {

// Binary code based on the ranking according to word frequencies.
class FrequencyCode : public BinaryCode {
  FrequencyCode(const FrequencyCode &) = delete;
  FrequencyCode(FrequencyCode &&) = delete;
  FrequencyCode & operator=(const FrequencyCode &) = delete;
  FrequencyCode & operator=(FrequencyCode &&) = delete;

public:
  // Constructs an empty object.
  FrequencyCode() {}

  // Constructs a new binary code object.
  //
  // Arguments:
  //   vocab: Vocabulary object for the target language.
  FrequencyCode(const Vocabulary & vocab);

  ~FrequencyCode() override {};

  std::vector<float> getCode(const unsigned id) const override;
  unsigned getID(const std::vector<float> & probs) const override;
  unsigned getNumBits() const override;

private:
  // Boost serialization interface
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive & ar, const unsigned) {
    ar & boost::serialization::base_object<BinaryCode>(*this);
    ar & wid_to_code_;
    ar & code_to_wid_;
    ar & vocab_size_;
    ar & num_bits_;
  }

  std::vector<unsigned> wid_to_code_;
  std::vector<unsigned> code_to_wid_;
  unsigned vocab_size_;
  unsigned num_bits_;
};

}  // namespace nmtkit

#endif  // NMTKIT_FREQUENCY_CODE_H_

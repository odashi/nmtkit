#ifndef NMTKIT_IDENTITY_ECC_H_
#define NMTKIT_IDENTITY_ECC_H_

#include <boost/serialization/base_object.hpp>
#include <nmtkit/error_correcting_code.h>

namespace nmtkit {

// ErrorCorrectingCode class for identity mapping.
class IdentityECC : public ErrorCorrectingCode {
  IdentityECC(const IdentityECC &) = delete;
  IdentityECC(IdentityECC &&) = delete;
  IdentityECC & operator=(const IdentityECC &) = delete;
  IdentityECC & operator=(IdentityECC &&) = delete;

public:
  IdentityECC() {}
  ~IdentityECC() override {}

  std::vector<bool> encode(
      const std::vector<bool> & original_bits) const override {
    return original_bits;
  }

  std::vector<float> decode(
      const std::vector<float> & encoded_probs) const override {
    return encoded_probs;
  }

  unsigned getNumBits(const unsigned original_num_bits) const override {
    return original_num_bits;
  }

private:
  // Boost serialization interface.
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive & ar, const unsigned) {
    ar & boost::serialization::base_object<ErrorCorrectingCode>(*this);
  }
};

}  // namespace nmtkit

NMTKIT_SERIALIZATION_DECL(nmtkit::IdentityECC);

#endif  // NMTKIT_IDENTITY_ECC_H_

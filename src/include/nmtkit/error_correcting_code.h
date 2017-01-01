#ifndef NMTKIT_ERROR_CORRECTING_CODE_H_
#define NMTKIT_ERROR_CORRECTING_CODE_H_

#include <vector>
#include <nmtkit/serialization_utils.h>

namespace nmtkit {

// Base class of error correcting code.
class ErrorCorrectingCode {
  ErrorCorrectingCode(const ErrorCorrectingCode &) = delete;
  ErrorCorrectingCode(ErrorCorrectingCode &&) = delete;
  ErrorCorrectingCode & operator=(const ErrorCorrectingCode &) = delete;
  ErrorCorrectingCode & operator=(ErrorCorrectingCode &&) = delete;

public:
  ErrorCorrectingCode() {}
  virtual ~ErrorCorrectingCode() {}

  // Calculates encoded bit array.
  //
  // Arguments:
  //   original_bits: Bit array to be encoded.
  //
  // Returns:
  //   Encoded bit array calculated from `original_bits`.
  virtual std::vector<bool> encode(
      const std::vector<bool> & original_bits) const = 0;

  // Retrieves original bit array.
  //
  // Arguments:
  //   encoded_probs: Encoded bit probabilities.
  //
  // Returns:
  //   Probability of each bit representing original bit array restored from
  //   `encoded_bits`.
  virtual std::vector<float> decode(
      const std::vector<float> & encoded_probs) const = 0;

  // Retrieves the number of bits in a encoded array.
  //
  // Arguments:
  //   original_num_bits: Number of bits in an original bit array.
  //
  // Returns:
  //   Number of bits in an encoded array.
  virtual unsigned getNumBits(const unsigned original_num_bits) const = 0;

private:
  // Boost serialization interface.
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive & ar, const unsigned) {}
};

}  // namespace nmtkit

NMTKIT_SERIALIZATION_DECL(nmtkit::ErrorCorrectingCode);

#endif  // NMTKIT_ERROR_CORRECTING_CODE_H_

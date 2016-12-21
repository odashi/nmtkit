#ifndef NMTKIT_BINARY_CODE_H_
#define NMTKIT_BINARY_CODE_H_

#include <vector>
#include <nmtkit/serialization_utils.h>

namespace nmtkit {

class BinaryCode {
  BinaryCode(const BinaryCode &) = delete;
  BinaryCode(BinaryCode &&) = delete;
  BinaryCode & operator=(const BinaryCode &) = delete;
  BinaryCode & operator=(BinaryCode &&) = delete;

public:
  BinaryCode() {}
  virtual ~BinaryCode() {}

  // Retrieves bit representation of given ID.
  //
  // Arguments:
  //   id: Target ID.
  //
  // Returns:
  //   List of bit probabilities.
  //   1.0 represents the bit becomes 1, and 0.0 represents 0.
  virtual std::vector<float> getCode(const unsigned id) const = 0;

  // Retrieves original ID by analysing bit probabilities.
  //
  // Arguments:
  //   probs: Target bit probabilities.
  //
  // Returns:
  //   The most probable ID.
  virtual unsigned getID(const std::vector<float> & probs) const = 0;

  // Retrieves the number of bits in a binary code.
  virtual unsigned getNumBits() const = 0;

private:
  // Boost serialization interface.
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive & ar, const unsigned) {}
};

}  // namespace nmtkit

NMTKIT_SERIALIZATION_DECL(nmtkit::BinaryCode);

#endif  // NMTKIT_BINARY_CODE_H_

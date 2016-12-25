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
  // Special word ID for invalid code.
  static const unsigned INVALID_CODE = 0xffffffffu;

  BinaryCode() {}
  virtual ~BinaryCode() {}

  // Retrieves bit representation of given word ID.
  //
  // Arguments:
  //   id: Target word ID.
  //
  // Returns:
  //   List of bits.
  virtual std::vector<bool> getCode(const unsigned id) const = 0;

  // Retrieves original ID by analysing a bit array.
  //
  // Arguments:
  //   code: Target bit array.
  //
  // Returns:
  //   Word ID corresponding to given bits, or INVALID_CODE if bits did not
  //   correspond to any word IDs.
  virtual unsigned getID(const std::vector<bool> & code) const = 0;

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

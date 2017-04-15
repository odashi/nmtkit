#ifndef NMTKIT_HUFFMAN_CODE_H_
#define NMTKIT_HUFFMAN_CODE_H_

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/vector.hpp>
#include <nmtkit/binary_code.h>
#include <nmtkit/vocabulary.h>
#include <nmtkit/serialization_utils.h>

namespace nmtkit {

class HuffmanCode : public BinaryCode {
  HuffmanCode(const HuffmanCode &) = delete;
  HuffmanCode(HuffmanCode &&) = delete;
  HuffmanCode & operator=(const HuffmanCode &) = delete;
  HuffmanCode & operator=(HuffmanCode &&) = delete;

  // Structure of internal tree.
  struct Node {
    unsigned freq;
    unsigned depth;
    int left;
    int right;
    std::vector<bool> code;
  private:
    // Bost serialization interface.
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive & ar, const unsigned) {
      ar & freq;
      ar & depth;
      ar & left;
      ar & right;
      ar & code;
    }
  };

public:
  // Constructs an empty object.
  HuffmanCode() {}

  // Constructs a new Huffman code object.
  //
  // Arguments:
  //   vocab: Vocabulary object for the target language.
  explicit HuffmanCode(const Vocabulary & vocab);

  ~HuffmanCode() override {}

  std::vector<bool> getCode(const unsigned id) const override;
  unsigned getID(const std::vector<bool> & code) const override;
  unsigned getNumBits() const override;

private:
  // Boost serialization interface.
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive & ar, const unsigned) {
    ar & boost::serialization::base_object<BinaryCode>(*this);
    ar & pool_;
  }

  std::vector<Node> pool_;
};

}  // namespace

NMTKIT_SERIALIZATION_DECL(nmtkit::HuffmanCode);

#endif  // NMTKIT_HUFFMAN_CODE_H_

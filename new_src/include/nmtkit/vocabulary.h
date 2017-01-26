#ifndef NMTKIT_VOCABULARY_H_
#define NMTKIT_VOCABULARY_H_

#include <boost/serialization/access.hpp>
#include <nmtkit/basic_types.h>
#include <nmtkit/serialization.h>
#include <vector>

namespace nmtkit {

// Abstract class of conversion methods between words and word IDs.
class Vocabulary {
  Vocabulary(const Vocabulary &) = delete;
  Vocabulary(Vocabulary &&) = delete;
  Vocabulary & operator=(const Vocabulary &) = delete;
  Vocabulary & operator=(Vocabulary &&) = delete;

 public:
  Vocabulary() {}
  virtual ~Vocabulary() {}

  // Retrieves a list of token IDs corresponding to given Sentence object.
  // This function may modify input sequences according to the policy of each
  // vocabulary, i.e., the length of given Sentence and returned vector may
  // differ.
  //
  // Arguments:
  //   sentence: A Sentence object.
  //
  // Returns:
  //   List of corresponding token IDs.
  virtual std::vector<unsigned> encode(const Sentence & sentence) const = 0;

  // Restores the Sentence object from given token IDs.
  // This function may return a Sentence object which have different number of
  // Token objects from the input vector, because of same reason of `encode()`.
  //
  // Arguments:
  //   ids: List of token IDs.
  //
  // Returns:
  //   Sentence object.
  virtual Sentence decode(const std::vector<unsigned> ids) const = 0;

  // Retrieves expected frequency of given token ID in the corpus.
  // Usually, this value would be calculated using the training corpus when
  // constructing actual vocabulary object.
  //
  // Arguments:
  //   id: A token ID.
  //
  // Returns:
  //   Expected frequency of given token ID in the corpus.
  virtual unsigned getFrequency(const unsigned id) const = 0;

  // Retrieves the size of the vocabulary.
  // Each token ID should satisfy that:
  //   0 <= (token ID) < Vocabulary::size()
  virtual unsigned size() const = 0;

 private:
  // Boost serialization interface.
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive & ar, const unsigned) {}  // NOLINT
};

}  // namespace nmtkit

NMTKIT_SERIALIZATION_DECL(nmtkit::Vocabulary);

#endif  // NMTKIT_VOCABULARY_H_

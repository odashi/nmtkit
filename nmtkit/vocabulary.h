#ifndef NMTKIT_VOCABULARY_H_
#define NMTKIT_VOCABULARY_H_

#include <string>
#include <vector>
#include <boost/serialization/access.hpp>
#include <nmtkit/serialization_utils.h>

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

  // Retrieves a word ID according to given word.
  //
  // Arguments:
  //   word: A word string.
  //
  // Returns:
  //   Corresponding word ID of `word`.
  virtual unsigned getID(const std::string & word) const = 0;

  // Retrieves actual word according to given word ID.
  //
  // Arguments:
  //   id: A word ID.
  //
  // Returns:
  //   Corresponding word string of `id`.
  virtual std::string getWord(const unsigned id) const = 0;

  // Retrieves frequency of given word in the corpus.
  //
  // Arguments:
  //   id: A word ID.
  //
  // Returns:
  //   A number representing the frequency of given word in the corpus.
  virtual unsigned getFrequency(const unsigned id) const = 0;

  // Converts a sentence into a list of word IDs.
  //
  // Arguments:
  //   sentence: A sentence string.
  //
  // Returns:
  //   List of word IDs that represents given sentence.
  virtual std::vector<unsigned> convertToIDs(
      const std::string & sentence) const = 0;

  // Converts a list of word IDs into a sentence.
  //
  // Arguments:
  //   word_ids: A list of word IDs.
  //
  // Returns:
  //   Generaed sentence string.
  virtual std::string convertToSentence(
      const std::vector<unsigned> & word_ids) const = 0;

  // Retrieves the size of the vocabulary.
  //
  // Returns:
  //   The size of the vocabulary.
  virtual unsigned size() const = 0;

private:
  // Boost serialization interface.
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive & ar, const unsigned) {}
};

}  // namespace nmtkit

NMTKIT_SERIALIZATION_DECL(nmtkit::Vocabulary);

#endif  // NMTKIT_VOCABULARY_H_

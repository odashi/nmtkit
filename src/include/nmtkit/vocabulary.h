#ifndef NMTKIT_VOCABULARY_H_
#define NMTKIT_VOCABULARY_H_

#include <map>
#include <string>
#include <vector>

namespace nmtkit {

class Vocabulary {
  Vocabulary() = delete;
  Vocabulary(const Vocabulary &) = delete;
  Vocabulary(Vocabulary &&) = delete;
  Vocabulary & operator=(const Vocabulary &) = delete;
  Vocabulary & operator=(Vocabulary &&) = delete;

public:
  // Loads existing vocabulary from a file.
  //
  // Arguments:
  //   vocab_filename: Location of the vocabulary file.
  explicit Vocabulary(const std::string & vocab_filename);

  // Analyzes a corpus and make a new vocabulary.
  //
  // Arguments:
  //   corpus_filename: Location of the corpus file to be analyzed.
  //   size: Size of the vocabulary.
  Vocabulary(const std::string & corpus_filename, unsigned size);

  // Saves the vocabulary to a file.
  //
  // Arguments:
  //   vocab_filename: Location of the vocabulary file.
  void save(const std::string & vocab_filename) const;

  // Retrieves a word ID according to given word.
  //
  // Arguments:
  //   word: A word string.
  //
  // Returns:
  //   Corresponding word ID of `word`.
  unsigned getID(const std::string & word) const;

  // Retrieves actual word according to given word ID.
  //
  // Arguments:
  //   id: A word ID.
  //
  // Returns:
  //   Corresponding word string of `id`.
  std::string getWord(unsigned id) const;

  // Converts a sentence into a list of word IDs.
  //
  // Arguments:
  //   sentence: A sentence string.
  //
  // Returns:
  //   List of word IDs that represents given sentence.
  std::vector<unsigned> convertToIDs(const std::string & sentence) const;

  // Retrieves the size of the vocabulary.
  //
  // Returns:
  //   The size of the vocabulary.
  unsigned size() const;

private:
  std::map<std::string, unsigned> stoi_;
  std::vector<std::string> itos_;
};

}  // namespace nmtkit

#endif  // NMTKIT_VOCABULARY_H_

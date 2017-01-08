#ifndef NMTKIT_CHARACTER_VOCABULARY_H_
#define NMTKIT_CHARACTER_VOCABULARY_H_

#include <map>
#include <string>
#include <vector>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <nmtkit/serialization_utils.h>
#include <nmtkit/vocabulary.h>

namespace nmtkit {

// Character-based vocabulary.
// Whitespaces are also encoded into their IDs.
// E.g.
//   input sentence:  "It works."
//   output word IDs: [ID("I"), ID("t"), ID(" "), ID("w"), ID("o"), ...]
class CharacterVocabulary : public Vocabulary {
  CharacterVocabulary(const CharacterVocabulary &) = delete;
  CharacterVocabulary(CharacterVocabulary &&) = delete;
  CharacterVocabulary & operator=(const CharacterVocabulary &) = delete;
  CharacterVocabulary & operator=(CharacterVocabulary &&) = delete;

public:
  // Initializes an empty vocabulary.
  CharacterVocabulary() {}

  // Analyzes a corpus and make a new vocabulary.
  //
  // Arguments:
  //   corpus_filename: Location of the corpus file to be analyzed.
  //   size: Size of the vocabulary.
  CharacterVocabulary(const std::string & corpus_filename, unsigned size);

  ~CharacterVocabulary() override {}

  unsigned getID(const std::string & word) const override;
  std::string getWord(const unsigned id) const override;
  unsigned getFrequency(const unsigned id) const override;
  std::vector<unsigned> convertToIDs(
      const std::string & sentence) const override;
  std::vector<std::string> convertToTokens(
      const std::string & sentence) const override;
  std::string convertToSentence(
      const std::vector<unsigned> & word_ids) const override;
  std::vector<std::string> convertToTokenizedSentence(
      const std::vector<unsigned> & word_ids) const override;
  unsigned size() const override;

private:
  // Boost serialization interface.
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive & ar, const unsigned) {
    ar & boost::serialization::base_object<Vocabulary>(*this);
    ar & stoi_;
    ar & itos_;
    ar & freq_;
  }

  std::map<std::string, unsigned> stoi_;
  std::vector<std::string> itos_;
  std::vector<unsigned> freq_;
};

}  // namespace nmtkit

NMTKIT_SERIALIZATION_DECL(nmtkit::CharacterVocabulary);

#endif  // NMTKIT_CHARACTER_VOCABULARY_H_

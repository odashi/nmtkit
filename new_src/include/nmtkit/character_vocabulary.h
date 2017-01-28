#ifndef NMTKIT_CHARACTER_VOCABULARY_H_
#define NMTKIT_CHARACTER_VOCABULARY_H_

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <nmtkit/reader.h>
#include <nmtkit/serialization.h>
#include <nmtkit/vocabulary.h>
#include <map>
#include <string>
#include <vector>

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
  //   reader: Reader object for the corpus to be analyzed.
  //   size: Size of the vocabulary.
  CharacterVocabulary(Reader * reader, const unsigned size);

  ~CharacterVocabulary() override {}

  std::vector<unsigned> convertToIDs(const Token & token) const override;
  std::vector<unsigned> convertToIDs(const Sentence & sentence) const override;
  Sentence convertToSentence(const std::vector<unsigned> & ids) const override;
  std::string getSurface(const unsigned id) const override;
  unsigned getFrequency(const unsigned id) const override;
  unsigned size() const override;

 private:
  // Boost serialization interface.
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive & ar, const unsigned) {  // NOLINT
    ar & boost::serialization::base_object<Vocabulary>(*this);
    ar & stoi_;
    ar & itos_;
    ar & freq_;
  }

  std::map<std::string, unsigned> stoi_;
  std::vector<std::string> itos_;
  std::vector<unsigned> freq_;

  // Storage for already calculated IDs.
  mutable std::map<std::string, std::vector<unsigned>> cache_;
};

}  // namespace nmtkit

NMTKIT_SERIALIZATION_DECL(nmtkit::CharacterVocabulary);

#endif  // NMTKIT_CHARACTER_VOCABULARY_H_

#ifndef NMTKIT_WORD_VOCABULARY_H_
#define NMTKIT_WORD_VOCABULARY_H_

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

// Word-based vocabulary.
// Each words are assumed to be separated by whitespaces.
// E.g.
//   input sentence:  "This is a test."
//   output word IDs: [ID("This"), ID("is"), ID("a"), ID("test.")]
class WordVocabulary : public Vocabulary {
  WordVocabulary(const WordVocabulary &) = delete;
  WordVocabulary(WordVocabulary &&) = delete;
  WordVocabulary & operator=(const WordVocabulary &) = delete;
  WordVocabulary & operator=(WordVocabulary &&) = delete;

 public:
  // Initializes an empty vocabulary.
  WordVocabulary() {}

  // Analyzes a corpus and make a new vocabulary.
  //
  // Arguments:
  //   reader: Reader object for the corpus to be analyzed.
  //   size: Size of the vocabulary.
  WordVocabulary(Reader * reader, const unsigned size);

  ~WordVocabulary() override {}

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
};

}  // namespace nmtkit

NMTKIT_SERIALIZATION_DECL(nmtkit::WordVocabulary);

#endif  // NMTKIT_WORD_VOCABULARY_H_

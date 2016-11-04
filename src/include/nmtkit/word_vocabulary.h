#ifndef NMTKIT_WORD_VOCABULARY_H_
#define NMTKIT_WORD_VOCABULARY_H_

#include <map>
#include <string>
#include <vector>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <nmtkit/serialization_utils.h>
#include <nmtkit/vocabulary.h>

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
  //   corpus_filename: Location of the corpus file to be analyzed.
  //   size: Size of the vocabulary.
  WordVocabulary(const std::string & corpus_filename, unsigned size);
  
  ~WordVocabulary() override {}

  unsigned getID(const std::string & word) const override;
  std::string getWord(unsigned id) const override;
  std::vector<unsigned> convertToIDs(
      const std::string & sentence) const override;
  unsigned size() const override;

private:
  // Boost serialization interface.
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive & ar, const unsigned) {
    ar & boost::serialization::base_object<Vocabulary>(*this);
    ar & stoi_;
    ar & itos_;
  }

  std::map<std::string, unsigned> stoi_;
  std::vector<std::string> itos_;
};

}  // namespace nmtkit

NMTKIT_SERIALIZATION_DECL(nmtkit::WordVocabulary);

#endif  // NMTKIT_WORD_VOCABULARY_H_

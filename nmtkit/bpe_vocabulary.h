#ifndef NMTKIT_BPE_VOCABULARY_H_
#define NMTKIT_BPE_VOCABULARY_H_

#include <map>
#include <string>
#include <vector>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <nmtkit/serialization_utils.h>
#include <nmtkit/vocabulary.h>

namespace nmtkit {

// Byte Pair Encoding vocabulary.
// For more details, see
// https://arxiv.org/abs/1508.07909
class BPEVocabulary : public Vocabulary {
  BPEVocabulary(const BPEVocabulary &) = delete;
  BPEVocabulary(BPEVocabulary &&) = delete;
  BPEVocabulary & operator=(const BPEVocabulary &) = delete;
  BPEVocabulary & operator=(BPEVocabulary &&) = delete;

public:
  // Initializes an empty vocabulary.
  BPEVocabulary() {}

  // Analyzes a corpus and make a new vocabulary.
  //
  // Arguments:
  //   corpus_filename: Location of the corpus file to be analyzed.
  //   size: Size of the vocabulary.
  BPEVocabulary(
      const std::string & corpus_filename,
      const unsigned size);

  ~BPEVocabulary() override {}

  unsigned getID(const std::string & word) const override;
  std::string getWord(const unsigned id) const override;
  unsigned getFrequency(const unsigned id) const override;
  std::vector<unsigned> convertToIDs(
      const std::string & sentence) const override;
  std::string convertToSentence(
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
    ar & bpe_codes_;
  }

  std::map<std::string, unsigned> stoi_;
  std::vector<std::string> itos_;
  std::vector<unsigned> freq_;
  std::map<std::pair<std::string, std::string>, unsigned> bpe_codes_;

  // Store BPE converted words
  // Converting words may take some time, so we store converted words that we observed once
  mutable std::map<std::string, std::vector<std::string>> bpe_cache_;
};

}  // namespace nmtkit

NMTKIT_SERIALIZATION_DECL(nmtkit::BPEVocabulary);

#endif  // NMTKIT_BPE_VOCABULARY_H_

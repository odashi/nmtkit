#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>

#include <fstream>
#include <string>
#include <vector>
#include <nmtkit/character_vocabulary.h>
#include <nmtkit/serialization_utils.h>

using std::ifstream;
using std::string;
using std::vector;

namespace {

template <class T>
void loadArchive(const string & filepath, T * obj) {
  ifstream ifs(filepath);
  boost::archive::binary_iarchive iar(ifs);
  iar >> *obj;
}

}  // namespace

BOOST_AUTO_TEST_SUITE(CharacterVocabularyTest)

BOOST_AUTO_TEST_CASE(CheckLoadFromVocabularyFile_En) {
  nmtkit::CharacterVocabulary vocab;
  ::loadArchive("data/small.en.char.vocab", &vocab);
  BOOST_CHECK_EQUAL(36, vocab.size());
  const vector<string> topk {
    "<unk>", "<s>", "</s>", " ", "e", "t", "o", "a"};
  for (unsigned i = 0; i < topk.size(); ++i) {
    BOOST_CHECK_EQUAL(i, vocab.getID(topk[i]));
    BOOST_CHECK_EQUAL(topk[i], vocab.getWord(i));
  }
  BOOST_CHECK_EQUAL(0, vocab.getID("unknown-word"));
  BOOST_CHECK_EQUAL(0, vocab.getFrequency(0));
  BOOST_CHECK_EQUAL(500, vocab.getFrequency(1));
  BOOST_CHECK_EQUAL(500, vocab.getFrequency(2));
  BOOST_CHECK_EQUAL(3371, vocab.getFrequency(3));
}

BOOST_AUTO_TEST_CASE(CheckLoadFromVocabularyFile_Ja) {
  nmtkit::CharacterVocabulary vocab;
  ::loadArchive("data/small.ja.char.vocab", &vocab);
  BOOST_CHECK_EQUAL(100, vocab.size());
  const vector<string> topk {
    "<unk>", "<s>", "</s>", " ", "。", "は", "い", "た"};
  for (unsigned i = 0; i < topk.size(); ++i) {
    BOOST_CHECK_EQUAL(i, vocab.getID(topk[i]));
    BOOST_CHECK_EQUAL(topk[i], vocab.getWord(i));
  }
  BOOST_CHECK_EQUAL(0, vocab.getID("未知語"));
  BOOST_CHECK_EQUAL(1405, vocab.getFrequency(0));
  BOOST_CHECK_EQUAL(500, vocab.getFrequency(1));
  BOOST_CHECK_EQUAL(500, vocab.getFrequency(2));
  BOOST_CHECK_EQUAL(5126, vocab.getFrequency(3));
}

BOOST_AUTO_TEST_CASE(CheckLoadFromCorpus_En) {
  nmtkit::CharacterVocabulary vocab("data/small.en.tok", 100);
  BOOST_CHECK_EQUAL(36, vocab.size());
  const vector<string> topk {
    "<unk>", "<s>", "</s>", " ", "e", "t", "o", "a"};
  for (unsigned i = 0; i < topk.size(); ++i) {
    BOOST_CHECK_EQUAL(i, vocab.getID(topk[i]));
    BOOST_CHECK_EQUAL(topk[i], vocab.getWord(i));
  }
  BOOST_CHECK_EQUAL(0, vocab.getID("unknown-word"));
  BOOST_CHECK_EQUAL(0, vocab.getFrequency(0));
  BOOST_CHECK_EQUAL(500, vocab.getFrequency(1));
  BOOST_CHECK_EQUAL(500, vocab.getFrequency(2));
  BOOST_CHECK_EQUAL(3371, vocab.getFrequency(3));
}

BOOST_AUTO_TEST_CASE(CheckLoadFromCorpus_Ja) {
  nmtkit::CharacterVocabulary vocab("data/small.ja.tok", 100);
  BOOST_CHECK_EQUAL(100, vocab.size());
  const vector<string> topk {
    "<unk>", "<s>", "</s>", " ", "。", "は", "い", "た"};
  for (unsigned i = 0; i < topk.size(); ++i) {
    BOOST_CHECK_EQUAL(i, vocab.getID(topk[i]));
    BOOST_CHECK_EQUAL(topk[i], vocab.getWord(i));
  }
  BOOST_CHECK_EQUAL(0, vocab.getID("未知語"));
  BOOST_CHECK_EQUAL(1405, vocab.getFrequency(0));
  BOOST_CHECK_EQUAL(500, vocab.getFrequency(1));
  BOOST_CHECK_EQUAL(500, vocab.getFrequency(2));
  BOOST_CHECK_EQUAL(5126, vocab.getFrequency(3));
}

BOOST_AUTO_TEST_CASE(CheckConvertingToIDs) {
  nmtkit::CharacterVocabulary vocab;
  ::loadArchive("data/small.en.char.vocab", &vocab);
  const vector<string> sentences {
    "anything that can go wrong , will go wrong .",
    "there is always light behind the clouds .",
    "and yet it moves .",
    "これ は 日本 語 の テスト 文 で す 。",
  };
  const vector<vector<unsigned>> expected {
    { 7, 11, 17,  5,  9,  8, 11, 21,  3,  5,  9,  7,  5,  3, 20,  7,
     11,  3, 21,  6,  3, 18, 12,  6, 11, 21,  3, 29,  3, 18,  8, 13,
     13,  3, 21,  6,  3, 18, 12,  6, 11, 21,  3, 14},
    { 5,  9,  4, 12,  4,  3,  8, 10,  3,  7, 13, 18,  7, 17, 10,  3,
     13,  8, 21,  9,  5,  3, 24,  4,  9,  8, 11, 15,  3,  5,  9,  4,
      3, 20, 13,  6, 16, 15, 10,  3, 14},
    { 7, 11, 15,  3, 17,  4,  5,  3,  8,  5,  3, 19,  6, 26,  4, 10,
      3, 14},
    { 0,  0,  3,  0,  3,  0,  0,  3,  0,  3,  0,  3,  0,  0,  0,  3,
      0,  3,  0,  3,  0,  3,  0},
  };

  for (unsigned i = 0; i < sentences.size(); ++i) {
    vector<unsigned> observed = vocab.convertToIDs(sentences[i]);
    BOOST_CHECK_EQUAL_COLLECTIONS(
        expected[i].begin(), expected[i].end(),
        observed.begin(), observed.end());
  }
}

BOOST_AUTO_TEST_CASE(CheckConvertingToSentence) {
  nmtkit::CharacterVocabulary vocab;
  ::loadArchive("data/small.en.char.vocab", &vocab);
  const vector<vector<unsigned>> word_ids {
    { 7, 11, 17,  5,  9,  8, 11, 21,  3,  5,  9,  7,  5,  3, 20,  7,
     11,  3, 21,  6,  3, 18, 12,  6, 11, 21,  3, 29,  3, 18,  8, 13,
     13,  3, 21,  6,  3, 18, 12,  6, 11, 21,  3, 14},
    { 5,  9,  4, 12,  4,  3,  8, 10,  3,  7, 13, 18,  7, 17, 10,  3,
     13,  8, 21,  9,  5,  3, 24,  4,  9,  8, 11, 15,  3,  5,  9,  4,
      3, 20, 13,  6, 16, 15, 10,  3, 14},
    { 7, 11, 15,  3, 17,  4,  5,  3,  8,  5,  3, 19,  6, 26,  4, 10,
      3, 14},
    { 0,  0,  3,  0,  3,  0,  0,  3,  0,  3,  0,  3,  0,  0,  0,  3,
      0,  3,  0,  3,  0,  3,  0},
  };
  const vector<string> expected {
    "anything that can go wrong , will go wrong .",
    "there is always light behind the clouds .",
    "and yet it moves .",
    "<unk><unk> <unk> <unk><unk> <unk> <unk> <unk><unk><unk> <unk>"
    " <unk> <unk> <unk>",
  };

  for (unsigned i = 0; i < word_ids.size(); ++i) {
    string observed = vocab.convertToSentence(word_ids[i]);
    BOOST_CHECK_EQUAL(expected[i], observed);
  }
}

BOOST_AUTO_TEST_SUITE_END()

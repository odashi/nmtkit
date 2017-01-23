#include "config.h"

#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>

#include <fstream>
#include <string>
#include <vector>
#include <nmtkit/serialization_utils.h>
#include <nmtkit/bpe_vocabulary.h>

using namespace std;

namespace {

template <class T>
void loadArchive(const string & filepath, T * obj) {
  ifstream ifs(filepath);
  boost::archive::binary_iarchive iar(ifs);
  iar >> *obj;
}

}  // namespace

BOOST_AUTO_TEST_SUITE(BPEVocabularyTest)

BOOST_AUTO_TEST_CASE(CheckLoadFromVocabularyFile_En) {
  nmtkit::BPEVocabulary vocab;
  ::loadArchive("data/small.en.bpe.vocab", &vocab);
  BOOST_CHECK_EQUAL(100, vocab.size());
  const vector<string> topk {
    "<unk>", "<s>", "</s>", "</w>", "e", "t", "o", "a"};
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
  nmtkit::BPEVocabulary vocab;
  ::loadArchive("data/small.ja.bpe.vocab", &vocab);
  BOOST_CHECK_EQUAL(1000, vocab.size());
  const vector<string> topk {
    "<unk>", "<s>", "</s>", "</w>", "。", "は", "い", "た"};
  for (unsigned i = 0; i < topk.size(); ++i) {
    BOOST_CHECK_EQUAL(i, vocab.getID(topk[i]));
    BOOST_CHECK_EQUAL(topk[i], vocab.getWord(i));
  }
  BOOST_CHECK_EQUAL(0, vocab.getID("未知語"));
  BOOST_CHECK_EQUAL(0, vocab.getFrequency(0));
  BOOST_CHECK_EQUAL(500, vocab.getFrequency(1));
  BOOST_CHECK_EQUAL(500, vocab.getFrequency(2));
  BOOST_CHECK_EQUAL(5126, vocab.getFrequency(3));
}

BOOST_AUTO_TEST_CASE(CheckLoadFromCorpus_En) {
  nmtkit::BPEVocabulary vocab("data/small.en.tok", 100);
  BOOST_CHECK_EQUAL(100, vocab.size());
  const vector<string> topk {
    "<unk>", "<s>", "</s>", "</w>", "e", "t", "o", "a"};
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
  nmtkit::BPEVocabulary vocab("data/small.ja.tok", 1000);
  BOOST_CHECK_EQUAL(1000, vocab.size());
  const vector<string> topk {
    "<unk>", "<s>", "</s>", "</w>", "。", "は", "い", "た"};
  for (unsigned i = 0; i < topk.size(); ++i) {
    BOOST_CHECK_EQUAL(i, vocab.getID(topk[i]));
    BOOST_CHECK_EQUAL(topk[i], vocab.getWord(i));
  }
  BOOST_CHECK_EQUAL(0, vocab.getID("未知語"));
  BOOST_CHECK_EQUAL(0, vocab.getFrequency(0));
  BOOST_CHECK_EQUAL(500, vocab.getFrequency(1));
  BOOST_CHECK_EQUAL(500, vocab.getFrequency(2));
  BOOST_CHECK_EQUAL(5126, vocab.getFrequency(3));
}

BOOST_AUTO_TEST_CASE(CheckConvertingToIDs) {
  nmtkit::BPEVocabulary vocab;
  ::loadArchive("data/small.en.bpe.vocab", &vocab);
  const vector<string> sentences {
    "anything that can go wrong , will go wrong .",
    "there is always light behind the clouds .",
    "and yet it moves .",
    "これ は 日本 語 の テスト 文 で す 。",
  };
  const vector<vector<unsigned>> expected {
    {77, 17, 40, 68, 40, 79, 85, 41, 21, 42, 18, 12, 66, 56,
     29, 3, 91, 73, 21, 42, 18, 12, 66, 56, 37},
    {40, 62, 36, 47, 7, 13, 67, 17, 39, 95, 21, 9, 38, 24, 4,
     9, 49, 43, 50, 20, 13, 44, 15, 39, 37},
    {77, 43, 17, 97, 82, 19, 6, 26, 4, 39, 37},
    {0, 0, 3, 0, 3, 0, 0, 3, 0, 3, 0, 3, 0, 0, 0, 3, 0, 3, 0,
     3, 0, 3, 0, 3}
  };

  for (unsigned i = 0; i < sentences.size(); ++i) {
    vector<unsigned> observed = vocab.convertToIDs(sentences[i]);
    BOOST_CHECK_EQUAL_COLLECTIONS(
        expected[i].begin(), expected[i].end(),
        observed.begin(), observed.end());
  }
}

BOOST_AUTO_TEST_CASE(CheckConvertingToSentence) {
  nmtkit::BPEVocabulary vocab;
  ::loadArchive("data/small.en.bpe.vocab", &vocab);
  const vector<vector<unsigned>> word_ids {
    {77, 17, 40, 68, 40, 79, 85, 41, 21, 42, 18, 12, 66, 56,
     29, 3, 91, 73, 21, 42, 18, 12, 66, 56, 37},
    {40, 62, 36, 47, 7, 13, 67, 17, 39, 95, 21, 9, 38, 24, 4,
     9, 49, 43, 50, 20, 13, 44, 15, 39, 37},
    {77, 43, 17, 97, 82, 19, 6, 26, 4, 39, 37},
    {0, 0, 3, 0, 3, 0, 0, 3, 0, 3, 0, 3, 0, 0, 0, 3, 0, 3, 0},
  };
  const vector<string> expected {
    "anything that can go wrong , will go wrong .",
    "there is always light behind the clouds .",
    "and yet it moves .",
    "<unk><unk> <unk> <unk><unk> <unk> <unk> <unk><unk><unk> <unk> <unk>",
  };

  for (unsigned i = 0; i < word_ids.size(); ++i) {
    string observed = vocab.convertToSentence(word_ids[i]);
    BOOST_CHECK_EQUAL(expected[i], observed);
  }
}

BOOST_AUTO_TEST_SUITE_END()

#include "config.h"

#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>

#include <fstream>
#include <string>
#include <vector>
#include <boost/archive/text_iarchive.hpp>
#include <nmtkit/word_vocabulary.h>

using namespace std;

namespace {

template <class T>
void loadArchive(const string & filepath, T * obj) {
  ifstream ifs(filepath);
  boost::archive::text_iarchive iar(ifs);
  iar >> *obj;
}

}  // namespace

BOOST_AUTO_TEST_SUITE(WordVocabularyTest)

BOOST_AUTO_TEST_CASE(CheckLoadFromVocabularyFile_En) {
  nmtkit::WordVocabulary vocab;
  ::loadArchive("data/small.en.vocab", &vocab);
  BOOST_CHECK_EQUAL(500, vocab.size());
  const vector<string> topk {
    "<unk>", "<s>", "</s>", ".", "the", "to", "i", "you"};
  for (unsigned i = 0; i < topk.size(); ++i) {
    BOOST_CHECK_EQUAL(i, vocab.getID(topk[i]));
    BOOST_CHECK_EQUAL(topk[i], vocab.getWord(i));
  }
  BOOST_CHECK_EQUAL(0, vocab.getID("unknown-word"));
  BOOST_CHECK_EQUAL("<unk>", vocab.getWord(500));
  BOOST_CHECK_EQUAL("<unk>", vocab.getWord(-1));
}

BOOST_AUTO_TEST_CASE(CheckLoadFromVocabularyFile_Ja) {
  nmtkit::WordVocabulary vocab;
  ::loadArchive("data/small.ja.vocab", &vocab);
  BOOST_CHECK_EQUAL(500, vocab.size());
  const vector<string> topk {
    "<unk>", "<s>", "</s>", "。", "は", "い", "に", "を"};
  for (unsigned i = 0; i < topk.size(); ++i) {
    BOOST_CHECK_EQUAL(i, vocab.getID(topk[i]));
    BOOST_CHECK_EQUAL(topk[i], vocab.getWord(i));
  }
  BOOST_CHECK_EQUAL(0, vocab.getID("未知語"));
  BOOST_CHECK_EQUAL("<unk>", vocab.getWord(500));
  BOOST_CHECK_EQUAL("<unk>", vocab.getWord(-1));
}

BOOST_AUTO_TEST_CASE(CheckLoadFromCorpus_En) {
  nmtkit::WordVocabulary vocab("data/small.en.tok", 100);
  BOOST_CHECK_EQUAL(100, vocab.size());
  const vector<string> topk {
    "<unk>", "<s>", "</s>", ".", "the", "to", "i", "you"};
  for (unsigned i = 0; i < topk.size(); ++i) {
    BOOST_CHECK_EQUAL(i, vocab.getID(topk[i]));
    BOOST_CHECK_EQUAL(topk[i], vocab.getWord(i));
  }
  BOOST_CHECK_EQUAL(0, vocab.getID("unknown-word"));
  BOOST_CHECK_EQUAL("<unk>", vocab.getWord(100));
  BOOST_CHECK_EQUAL("<unk>", vocab.getWord(-1));
}

BOOST_AUTO_TEST_CASE(CheckLoadFromCorpus_Ja) {
  nmtkit::WordVocabulary vocab("data/small.ja.tok", 100);
  BOOST_CHECK_EQUAL(100, vocab.size());
  const vector<string> topk {
    "<unk>", "<s>", "</s>", "。", "は", "い", "に", "を"};
  for (unsigned i = 0; i < topk.size(); ++i) {
    BOOST_CHECK_EQUAL(i, vocab.getID(topk[i]));
    BOOST_CHECK_EQUAL(topk[i], vocab.getWord(i));
  }
  BOOST_CHECK_EQUAL(0, vocab.getID("未知語"));
  BOOST_CHECK_EQUAL("<unk>", vocab.getWord(100));
  BOOST_CHECK_EQUAL("<unk>", vocab.getWord(-1));
}

BOOST_AUTO_TEST_CASE(CheckConvertingToIDs) {
  nmtkit::WordVocabulary vocab;
  ::loadArchive("data/small.en.vocab", &vocab);
  const vector<string> sentences {
    "anything that can go wrong , will go wrong .",
    "there is always light behind the clouds .",
    "and yet it moves .",
    "これ は 日本 語 の テスト 文 で す 。",
  };
  const vector<vector<unsigned>> expected {
    {0, 20, 41, 45, 134, 31, 37, 45, 134, 3},
    {39, 9, 85, 0, 400, 4, 0, 3},
    {56, 183, 16, 0, 3},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  };

  for (unsigned i = 0; i < sentences.size(); ++i) {
    vector<unsigned> observed = vocab.convertToIDs(sentences[i]);
    BOOST_CHECK_EQUAL_COLLECTIONS(
        expected[i].begin(), expected[i].end(),
        observed.begin(), observed.end());
  }
}

BOOST_AUTO_TEST_CASE(CheckConvertingToSentence) {
  nmtkit::WordVocabulary vocab;
  ::loadArchive("data/small.en.vocab", &vocab);
  const vector<vector<unsigned>> word_ids {
    {0, 20, 41, 45, 134, 31, 37, 45, 134, 3},
    {39, 9, 85, 0, 400, 4, 0, 3},
    {56, 183, 16, 0, 3},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  };
  const vector<string> expected {
    "<unk> that can go wrong , will go wrong .",
    "there is always <unk> behind the <unk> .",
    "and yet it <unk> .",
    "<unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk>",
  };

  for (unsigned i = 0; i < word_ids.size(); ++i) {
    string observed = vocab.convertToSentence(word_ids[i]);
    BOOST_CHECK_EQUAL(expected[i], observed);
  }
}

BOOST_AUTO_TEST_SUITE_END()

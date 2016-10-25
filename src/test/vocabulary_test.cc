#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>

#include <vector>
#include <string>
#include <nmtkit/vocabulary.h>

using namespace std;

BOOST_AUTO_TEST_SUITE(VocabularyTest)

BOOST_AUTO_TEST_CASE(CheckLoadFromVocabularyFile_En) {
  nmtkit::Vocabulary vocab("data/small.en.vocab");
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
  nmtkit::Vocabulary vocab("data/small.ja.vocab");
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
  nmtkit::Vocabulary vocab("data/small.en.tok", 100);
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
  nmtkit::Vocabulary vocab("data/small.ja.tok", 100);
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

BOOST_AUTO_TEST_SUITE_END()

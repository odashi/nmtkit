#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>

#include <nmtkit/vocabulary.h>

BOOST_AUTO_TEST_SUITE(VocabularyTest)

BOOST_AUTO_TEST_CASE(CheckLoadFromVocabularyFile) {
  NMTKit::Vocabulary vocab("data/sample.vocab");
  BOOST_CHECK_EQUAL(5, vocab.size());
  BOOST_CHECK_EQUAL(0, vocab.getID("<unk>"));
  BOOST_CHECK_EQUAL(1, vocab.getID("<s>"));
  BOOST_CHECK_EQUAL(2, vocab.getID("</s>"));
  BOOST_CHECK_EQUAL(3, vocab.getID("foo"));
  BOOST_CHECK_EQUAL(4, vocab.getID("bar"));
  BOOST_CHECK_EQUAL(0, vocab.getID("baz"));
  BOOST_CHECK_EQUAL("<unk>", vocab.getWord(0));
  BOOST_CHECK_EQUAL("<s>", vocab.getWord(1));
  BOOST_CHECK_EQUAL("</s>", vocab.getWord(2));
  BOOST_CHECK_EQUAL("foo", vocab.getWord(3));
  BOOST_CHECK_EQUAL("bar", vocab.getWord(4));
  BOOST_CHECK_EQUAL("<unk>", vocab.getWord(5));
  BOOST_CHECK_EQUAL("<unk>", vocab.getWord(-1));
}

BOOST_AUTO_TEST_CASE(CheckLoadFromCorpus) {
  NMTKit::Vocabulary vocab("data/lipsum.tok", 100);
  BOOST_CHECK_EQUAL(100, vocab.size());
  BOOST_CHECK_EQUAL(0, vocab.getID("<unk>"));
  BOOST_CHECK_EQUAL(1, vocab.getID("<s>"));
  BOOST_CHECK_EQUAL(2, vocab.getID("</s>"));
  BOOST_CHECK_EQUAL(3, vocab.getID("."));
  BOOST_CHECK_EQUAL(4, vocab.getID(","));
  BOOST_CHECK_EQUAL(5, vocab.getID("ne"));
  BOOST_CHECK_EQUAL(10, vocab.getID("no"));
  BOOST_CHECK_EQUAL(15, vocab.getID("cu"));
  BOOST_CHECK_EQUAL(20, vocab.getID("vim"));
  BOOST_CHECK_EQUAL(0, vocab.getID("unknown-word"));
  BOOST_CHECK_EQUAL("<unk>", vocab.getWord(0));
  BOOST_CHECK_EQUAL("<s>", vocab.getWord(1));
  BOOST_CHECK_EQUAL("</s>", vocab.getWord(2));
  BOOST_CHECK_EQUAL(".", vocab.getWord(3));
  BOOST_CHECK_EQUAL(",", vocab.getWord(4));
  BOOST_CHECK_EQUAL("ne", vocab.getWord(5));
  BOOST_CHECK_EQUAL("no", vocab.getWord(10));
  BOOST_CHECK_EQUAL("cu", vocab.getWord(15));
  BOOST_CHECK_EQUAL("vim", vocab.getWord(20));
  BOOST_CHECK_EQUAL("<unk>", vocab.getWord(100));
  BOOST_CHECK_EQUAL("<unk>", vocab.getWord(-1));
}

BOOST_AUTO_TEST_SUITE_END()

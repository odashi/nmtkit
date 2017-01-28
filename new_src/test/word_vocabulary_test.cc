#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>

#include <config.h>
#include <nmtkit/serialization.h>
#include <nmtkit/simple_text_reader.h>
#include <nmtkit/word_vocabulary.h>
#include <fstream>
#include <string>
#include <vector>

using std::string;
using std::vector;
using nmtkit::Token;
using nmtkit::Sentence;
using nmtkit::SimpleTextReader;
using nmtkit::WordVocabulary;

namespace {

template <class T>
void loadArchive(const string & filepath, T * obj) {
  std::ifstream ifs(filepath);
  boost::archive::binary_iarchive iar(ifs);
  iar >> *obj;
}

}  // namespace

BOOST_AUTO_TEST_SUITE(WordVocabularyTest)

BOOST_AUTO_TEST_CASE(CheckLoadFromVocabularyFile_En) {
  const vector<string> topk {
    "<unk>", "<s>", "</s>", ".", "the", "to", "i", "you"
  };

  WordVocabulary vocab;
  ::loadArchive("data/small.en.vocab", &vocab);
  BOOST_CHECK_EQUAL(500, vocab.size());

  for (unsigned i = 0; i < topk.size(); ++i) {
    const vector<unsigned> ids = vocab.convertToIDs(Token(topk[i]));
    BOOST_CHECK_EQUAL(1, ids.size());
    BOOST_CHECK_EQUAL(i, ids[0]);
    BOOST_CHECK_EQUAL(topk[i], vocab.getSurface(i));
  }

  const vector<unsigned> unk_ids = vocab.convertToIDs(Token("unknown-word"));
  BOOST_CHECK_EQUAL(1, unk_ids.size());
  BOOST_CHECK_EQUAL(0, unk_ids[0]);

  BOOST_CHECK_EQUAL(335, vocab.getFrequency(0));
  BOOST_CHECK_EQUAL(500, vocab.getFrequency(1));
  BOOST_CHECK_EQUAL(500, vocab.getFrequency(2));
  BOOST_CHECK_EQUAL(443, vocab.getFrequency(3));
}

BOOST_AUTO_TEST_CASE(CheckLoadFromVocabularyFile_Ja) {
  const vector<string> topk {
    "<unk>", "<s>", "</s>", "。", "は", "い", "に", "を"
  };

  WordVocabulary vocab;
  ::loadArchive("data/small.ja.vocab", &vocab);
  BOOST_CHECK_EQUAL(500, vocab.size());

  for (unsigned i = 0; i < topk.size(); ++i) {
    const vector<unsigned> ids = vocab.convertToIDs(Token(topk[i]));
    BOOST_CHECK_EQUAL(1, ids.size());
    BOOST_CHECK_EQUAL(i, ids[0]);
    BOOST_CHECK_EQUAL(topk[i], vocab.getSurface(i));
  }

  const vector<unsigned> unk_ids = vocab.convertToIDs(Token("未知語"));
  BOOST_CHECK_EQUAL(1, unk_ids.size());
  BOOST_CHECK_EQUAL(0, unk_ids[0]);

  BOOST_CHECK_EQUAL(423, vocab.getFrequency(0));
  BOOST_CHECK_EQUAL(500, vocab.getFrequency(1));
  BOOST_CHECK_EQUAL(500, vocab.getFrequency(2));
  BOOST_CHECK_EQUAL(498, vocab.getFrequency(3));
}

BOOST_AUTO_TEST_CASE(CheckLoadFromCorpus_En) {
  const vector<string> topk {
    "<unk>", "<s>", "</s>", ".", "the", "to", "i", "you"
  };

  SimpleTextReader reader("data/small.en.tok");
  WordVocabulary vocab(&reader, 100);
  BOOST_CHECK_EQUAL(100, vocab.size());

  for (unsigned i = 0; i < topk.size(); ++i) {
    const vector<unsigned> ids = vocab.convertToIDs(Token { topk[i] });
    BOOST_CHECK_EQUAL(1, ids.size());
    BOOST_CHECK_EQUAL(i, ids[0]);
    BOOST_CHECK_EQUAL(topk[i], vocab.getSurface(i));
  }

  const vector<unsigned> unk_ids = vocab.convertToIDs(Token { "unknown-word" });
  BOOST_CHECK_EQUAL(1, unk_ids.size());
  BOOST_CHECK_EQUAL(0, unk_ids[0]);

  BOOST_CHECK_EQUAL(1327, vocab.getFrequency(0));
  BOOST_CHECK_EQUAL(500, vocab.getFrequency(1));
  BOOST_CHECK_EQUAL(500, vocab.getFrequency(2));
  BOOST_CHECK_EQUAL(443, vocab.getFrequency(3));
}

BOOST_AUTO_TEST_CASE(CheckLoadFromCorpus_Ja) {
  const vector<string> topk {
    "<unk>", "<s>", "</s>", "。", "は", "い", "に", "を"
  };

  SimpleTextReader reader("data/small.ja.tok");
  WordVocabulary vocab(&reader, 100);
  BOOST_CHECK_EQUAL(100, vocab.size());

  for (unsigned i = 0; i < topk.size(); ++i) {
    const vector<unsigned> ids = vocab.convertToIDs(Token { topk[i] });
    BOOST_CHECK_EQUAL(1, ids.size());
    BOOST_CHECK_EQUAL(i, ids[0]);
    BOOST_CHECK_EQUAL(topk[i], vocab.getSurface(i));
  }

  const vector<unsigned> unk_ids = vocab.convertToIDs(Token { "未知語" });
  BOOST_CHECK_EQUAL(1, unk_ids.size());
  BOOST_CHECK_EQUAL(0, unk_ids[0]);

  BOOST_CHECK_EQUAL(1319, vocab.getFrequency(0));
  BOOST_CHECK_EQUAL(500, vocab.getFrequency(1));
  BOOST_CHECK_EQUAL(500, vocab.getFrequency(2));
  BOOST_CHECK_EQUAL(498, vocab.getFrequency(3));
}

BOOST_AUTO_TEST_CASE(CheckConvertingToIDs) {
  const vector<Sentence> sentences {
    Sentence({
        "anything", "that", "can", "go", "wrong", ",", "will", "go", "wrong",
        "."}),
    Sentence({
        "there", "is", "always", "light", "behind", "the", "clouds", "."}),
    Sentence({
        "and", "yet", "it", "moves", "."}),
    Sentence({
        "これ", "は", "日本", "語", "の", "テスト", "文", "で", "す", "。"}),
  };
  const vector<vector<unsigned>> expected {
    {0, 20, 41, 45, 134, 31, 37, 45, 134, 3},
    {39, 9, 85, 0, 400, 4, 0, 3},
    {56, 183, 16, 0, 3},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  };

  WordVocabulary vocab;
  ::loadArchive("data/small.en.vocab", &vocab);

  for (unsigned i = 0; i < sentences.size(); ++i) {
    const vector<unsigned> observed = vocab.convertToIDs(sentences[i]);
    BOOST_CHECK_EQUAL_COLLECTIONS(
        expected[i].begin(), expected[i].end(),
        observed.begin(), observed.end());
  }
}

BOOST_AUTO_TEST_CASE(CheckConvertingToSentence) {
  const vector<vector<unsigned>> word_ids {
    {0, 20, 41, 45, 134, 31, 37, 45, 134, 3},
    {39, 9, 85, 0, 400, 4, 0, 3},
    {56, 183, 16, 0, 3},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  };
  const vector<Sentence> expected {
    Sentence({
        "<unk>", "that", "can", "go", "wrong", ",", "will", "go", "wrong",
        "."}),
    Sentence({
        "there", "is", "always", "<unk>", "behind", "the", "<unk>", "."}),
    Sentence({
        "and", "yet", "it", "<unk>", "."}),
    Sentence({
        "<unk>", "<unk>", "<unk>", "<unk>", "<unk>", "<unk>", "<unk>", "<unk>",
        "<unk>", "<unk>"}),
  };

  WordVocabulary vocab;
  ::loadArchive("data/small.en.vocab", &vocab);

  for (unsigned i = 0; i < word_ids.size(); ++i) {
    Sentence observed = vocab.convertToSentence(word_ids[i]);
    BOOST_CHECK_EQUAL(expected[i], observed);
  }
}

BOOST_AUTO_TEST_SUITE_END()

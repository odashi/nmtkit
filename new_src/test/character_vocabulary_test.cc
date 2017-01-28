#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>

#include <config.h>
#include <nmtkit/character_vocabulary.h>
#include <nmtkit/serialization.h>
#include <nmtkit/simple_text_reader.h>
#include <nmtkit/unicode.h>
#include <fstream>
#include <string>
#include <vector>

using std::string;
using std::vector;
using nmtkit::Token;
using nmtkit::Sentence;
using nmtkit::SimpleTextReader;
using nmtkit::CharacterVocabulary;
using nmtkit::UTF8;

namespace {

template <class T>
void loadArchive(const string & filepath, T * obj) {
  std::ifstream ifs(filepath);
  boost::archive::binary_iarchive iar(ifs);
  iar >> *obj;
}

}  // namespace

BOOST_AUTO_TEST_SUITE(CharacterVocabularyTest)

BOOST_AUTO_TEST_CASE(CheckLoadFromVocabularyFile_En) {
  const vector<string> topk {
    "<unk>", "<s>", "</s>", "<sp>", "e", "t", "o", "a"
  };

  CharacterVocabulary vocab;
  ::loadArchive("data/small.en.char.vocab", &vocab);
  BOOST_CHECK_EQUAL(36, vocab.size());

  for (unsigned i = 0; i < topk.size(); ++i) {
    const vector<string> letters = UTF8::getLetters(topk[i]);
    const vector<unsigned> ids = vocab.convertToIDs(Token(topk[i]));
    BOOST_CHECK_EQUAL(letters.size(), ids.size());
    if (letters.size() == 1) {
      BOOST_CHECK_EQUAL(i, ids[0]);
    }
    BOOST_CHECK_EQUAL(topk[i], vocab.getSurface(i));
  }

  const vector<unsigned> unk_ids = vocab.convertToIDs(Token("あ"));
  BOOST_CHECK_EQUAL(1, unk_ids.size());
  BOOST_CHECK_EQUAL(0, unk_ids[0]);

  BOOST_CHECK_EQUAL(0, vocab.getFrequency(0));
  BOOST_CHECK_EQUAL(500, vocab.getFrequency(1));
  BOOST_CHECK_EQUAL(500, vocab.getFrequency(2));
  BOOST_CHECK_EQUAL(3371, vocab.getFrequency(3));
}

BOOST_AUTO_TEST_CASE(CheckLoadFromVocabularyFile_Ja) {
  const vector<string> topk {
    "<unk>", "<s>", "</s>", "<sp>", "。", "は", "い", "た"
  };

  CharacterVocabulary vocab;
  ::loadArchive("data/small.ja.char.vocab", &vocab);
  BOOST_CHECK_EQUAL(100, vocab.size());

  for (unsigned i = 0; i < topk.size(); ++i) {
    const vector<string> letters = UTF8::getLetters(topk[i]);
    const vector<unsigned> ids = vocab.convertToIDs(Token(topk[i]));
    BOOST_CHECK_EQUAL(letters.size(), ids.size());
    if (letters.size() == 1) {
      BOOST_CHECK_EQUAL(i, ids[0]);
    }
    BOOST_CHECK_EQUAL(topk[i], vocab.getSurface(i));
  }

  const vector<unsigned> unk_ids = vocab.convertToIDs(Token("a"));
  BOOST_CHECK_EQUAL(1, unk_ids.size());
  BOOST_CHECK_EQUAL(0, unk_ids[0]);

  BOOST_CHECK_EQUAL(1405, vocab.getFrequency(0));
  BOOST_CHECK_EQUAL(500, vocab.getFrequency(1));
  BOOST_CHECK_EQUAL(500, vocab.getFrequency(2));
  BOOST_CHECK_EQUAL(5126, vocab.getFrequency(3));
}

BOOST_AUTO_TEST_CASE(CheckLoadFromCorpus_En) {
  const vector<string> topk {
    "<unk>", "<s>", "</s>", "<sp>", "e", "t", "o", "a"
  };

  SimpleTextReader reader("data/small.en.tok");
  CharacterVocabulary vocab(&reader, 100);
  BOOST_CHECK_EQUAL(36, vocab.size());

  for (unsigned i = 0; i < topk.size(); ++i) {
    const vector<string> letters = UTF8::getLetters(topk[i]);
    const vector<unsigned> ids = vocab.convertToIDs(Token(topk[i]));
    BOOST_CHECK_EQUAL(letters.size(), ids.size());
    if (letters.size() == 1) {
      BOOST_CHECK_EQUAL(i, ids[0]);
    }
    BOOST_CHECK_EQUAL(topk[i], vocab.getSurface(i));
  }

  const vector<unsigned> unk_ids = vocab.convertToIDs(Token("あ"));
  BOOST_CHECK_EQUAL(1, unk_ids.size());
  BOOST_CHECK_EQUAL(0, unk_ids[0]);

  BOOST_CHECK_EQUAL(0, vocab.getFrequency(0));
  BOOST_CHECK_EQUAL(500, vocab.getFrequency(1));
  BOOST_CHECK_EQUAL(500, vocab.getFrequency(2));
  BOOST_CHECK_EQUAL(3371, vocab.getFrequency(3));
}

BOOST_AUTO_TEST_CASE(CheckLoadFromCorpus_Ja) {
  const vector<string> topk {
    "<unk>", "<s>", "</s>", "<sp>", "。", "は", "い", "た"
  };

  SimpleTextReader reader("data/small.ja.tok");
  CharacterVocabulary vocab(&reader, 100);
  BOOST_CHECK_EQUAL(100, vocab.size());

  for (unsigned i = 0; i < topk.size(); ++i) {
    const vector<string> letters = UTF8::getLetters(topk[i]);
    const vector<unsigned> ids = vocab.convertToIDs(Token(topk[i]));
    BOOST_CHECK_EQUAL(letters.size(), ids.size());
    if (letters.size() == 1) {
      BOOST_CHECK_EQUAL(i, ids[0]);
    }
    BOOST_CHECK_EQUAL(topk[i], vocab.getSurface(i));
  }

  const vector<unsigned> unk_ids = vocab.convertToIDs(Token("a"));
  BOOST_CHECK_EQUAL(1, unk_ids.size());
  BOOST_CHECK_EQUAL(0, unk_ids[0]);

  BOOST_CHECK_EQUAL(1405, vocab.getFrequency(0));
  BOOST_CHECK_EQUAL(500, vocab.getFrequency(1));
  BOOST_CHECK_EQUAL(500, vocab.getFrequency(2));
  BOOST_CHECK_EQUAL(5126, vocab.getFrequency(3));
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

  CharacterVocabulary vocab;
  ::loadArchive("data/small.en.char.vocab", &vocab);

  for (unsigned i = 0; i < sentences.size(); ++i) {
    const vector<unsigned> observed = vocab.convertToIDs(sentences[i]);
    BOOST_CHECK_EQUAL_COLLECTIONS(
        expected[i].begin(), expected[i].end(),
        observed.begin(), observed.end());
  }
}

BOOST_AUTO_TEST_CASE(CheckConvertingToSentence) {
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
  const vector<Sentence> expected {
    Sentence({
        "anything", "that", "can", "go", "wrong", ",", "will", "go", "wrong",
        "."}),
    Sentence({
        "there", "is", "always", "light", "behind", "the", "clouds", "."}),
    Sentence({
        "and", "yet", "it", "moves", "."}),
    Sentence({
        "<unk><unk>", "<unk>", "<unk><unk>", "<unk>", "<unk>",
        "<unk><unk><unk>", "<unk>", "<unk>", "<unk>", "<unk>"}),
  };

  CharacterVocabulary vocab;
  ::loadArchive("data/small.en.char.vocab", &vocab);

  for (unsigned i = 0; i < word_ids.size(); ++i) {
    Sentence observed = vocab.convertToSentence(word_ids[i]);
    BOOST_CHECK_EQUAL(expected[i], observed);
  }
}

BOOST_AUTO_TEST_SUITE_END()

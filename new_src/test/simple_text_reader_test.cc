#include "config.h"

#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>

#include <fstream>
#include <string>
#include <vector>
#include <nmtkit/basic_types.h>
#include <nmtkit/simple_text_reader.h>

using std::string;
using std::vector;
using nmtkit::SentencePair;
using nmtkit::SimpleTextReader;

BOOST_AUTO_TEST_SUITE(SimpleTextReaderTest)

BOOST_AUTO_TEST_CASE(CheckReadingSrc) {
  const vector<vector<string>> expected {
    { "i", "can", "'t", "tell" ,"who", "will", "arrive", "first", "." },
    { "many", "animals", "have", "been", "destroyed", "by", "men", "." },
    { "i", "'m", "in", "the", "tennis", "club", "." },
    { "emi", "looks", "happy", "." },
    { "please", "bear", "this", "fact", "in", "mind", "." },
  };

  SimpleTextReader reader("data/small.en.tok", "");
  SentencePair sp;

  // Checks read data.
  for (unsigned i = 0; i < expected.size(); ++i) {
    BOOST_CHECK(reader.read(&sp));

    BOOST_CHECK(sp.source.features.empty());
    BOOST_CHECK(sp.target.tokens.empty());
    BOOST_CHECK(sp.target.features.empty());
    BOOST_CHECK(sp.features.empty());

    BOOST_CHECK_EQUAL(expected[i].size(), sp.source.tokens.size());
    for (unsigned j = 0; j < expected[i].size(); ++j) {
      BOOST_CHECK_EQUAL(expected[i][j], sp.source.tokens[j].surface);
      BOOST_CHECK(sp.source.tokens[j].features.empty());
    }
  }

  // Checks number of data.
  unsigned n = expected.size();
  while (reader.read(&sp)) ++n;
  BOOST_CHECK_EQUAL(500, n);
}

BOOST_AUTO_TEST_CASE(CheckReadingTrg) {
  const vector<vector<string>> expected {
    { "誰", "が", "一番", "に", "着", "く", "か", "私", "に", "は", "分か",
      "り", "ま", "せ", "ん", "。" },
    { "多く", "の", "動物", "が", "人間", "に", "よ", "っ", "て", "滅ぼ", "さ",
      "れ", "た", "。" },
    { "私", "は", "テニス", "部員", "で", "す", "。" },
    { "エミ", "は", "幸せ", "そう", "に", "見え", "ま", "す", "。" },
    { "この", "事実", "を", "心", "に", "留め", "て", "お", "い", "て", "下さ",
      "い", "。" },
  };

  SimpleTextReader reader("", "data/small.ja.tok");
  SentencePair sp;

  // Checks read data.
  for (unsigned i = 0; i < expected.size(); ++i) {
    BOOST_CHECK(reader.read(&sp));

    BOOST_CHECK(sp.source.tokens.empty());
    BOOST_CHECK(sp.source.features.empty());
    BOOST_CHECK(sp.target.features.empty());
    BOOST_CHECK(sp.features.empty());

    BOOST_CHECK_EQUAL(expected[i].size(), sp.target.tokens.size());
    for (unsigned j = 0; j < expected[i].size(); ++j) {
      BOOST_CHECK_EQUAL(expected[i][j], sp.target.tokens[j].surface);
      BOOST_CHECK(sp.target.tokens[j].features.empty());
    }
  }

  // Checks number of data.
  unsigned n = expected.size();
  while (reader.read(&sp)) ++n;
  BOOST_CHECK_EQUAL(500, n);
}

BOOST_AUTO_TEST_CASE(CheckReadingSrcTrg) {
  const vector<vector<string>> expected_src {
    { "i", "can", "'t", "tell" ,"who", "will", "arrive", "first", "." },
    { "many", "animals", "have", "been", "destroyed", "by", "men", "." },
    { "i", "'m", "in", "the", "tennis", "club", "." },
    { "emi", "looks", "happy", "." },
    { "please", "bear", "this", "fact", "in", "mind", "." },
  };
  const vector<vector<string>> expected_trg {
    { "誰", "が", "一番", "に", "着", "く", "か", "私", "に", "は", "分か",
      "り", "ま", "せ", "ん", "。" },
    { "多く", "の", "動物", "が", "人間", "に", "よ", "っ", "て", "滅ぼ", "さ",
      "れ", "た", "。" },
    { "私", "は", "テニス", "部員", "で", "す", "。" },
    { "エミ", "は", "幸せ", "そう", "に", "見え", "ま", "す", "。" },
    { "この", "事実", "を", "心", "に", "留め", "て", "お", "い", "て", "下さ",
      "い", "。" },
  };

  SimpleTextReader reader("data/small.en.tok", "data/small.ja.tok");
  SentencePair sp;

  // Checks read data.
  for (unsigned i = 0; i < expected_src.size(); ++i) {
    BOOST_CHECK(reader.read(&sp));

    BOOST_CHECK(sp.source.features.empty());
    BOOST_CHECK(sp.target.features.empty());
    BOOST_CHECK(sp.features.empty());

    BOOST_CHECK_EQUAL(expected_src[i].size(), sp.source.tokens.size());
    for (unsigned j = 0; j < expected_src[i].size(); ++j) {
      BOOST_CHECK_EQUAL(expected_src[i][j], sp.source.tokens[j].surface);
      BOOST_CHECK(sp.source.tokens[j].features.empty());
    }

    BOOST_CHECK_EQUAL(expected_trg[i].size(), sp.target.tokens.size());
    for (unsigned j = 0; j < expected_trg[i].size(); ++j) {
      BOOST_CHECK_EQUAL(expected_trg[i][j], sp.target.tokens[j].surface);
      BOOST_CHECK(sp.target.tokens[j].features.empty());
    }
  }

  // Checks number of data.
  unsigned n = expected_src.size();
  while (reader.read(&sp)) ++n;
  BOOST_CHECK_EQUAL(500, n);
}

BOOST_AUTO_TEST_SUITE_END()


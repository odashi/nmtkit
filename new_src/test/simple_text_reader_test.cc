#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>

#include <config.h>
#include <nmtkit/basic_types.h>
#include <nmtkit/simple_text_reader.h>
#include <fstream>
#include <string>
#include <vector>

using std::string;
using std::vector;
using nmtkit::Sentence;
using nmtkit::SimpleTextReader;

BOOST_AUTO_TEST_SUITE(SimpleTextReaderTest)

BOOST_AUTO_TEST_CASE(CheckReading_En) {
  const vector<vector<string>> expected {
    { "i", "can", "'t", "tell", "who", "will", "arrive", "first", "." },
    { "many", "animals", "have", "been", "destroyed", "by", "men", "." },
    { "i", "'m", "in", "the", "tennis", "club", "." },
    { "emi", "looks", "happy", "." },
    { "please", "bear", "this", "fact", "in", "mind", "." },
  };

  SimpleTextReader reader("data/small.en.tok");
  Sentence sent;

  // Checks read data.
  for (unsigned i = 0; i < expected.size(); ++i) {
    BOOST_CHECK(reader.read(&sent));
    BOOST_CHECK(sent.features.empty());
    BOOST_CHECK_EQUAL(expected[i].size(), sent.tokens.size());
    for (unsigned j = 0; j < expected[i].size(); ++j) {
      BOOST_CHECK_EQUAL(expected[i][j], sent.tokens[j].surface);
      BOOST_CHECK(sent.tokens[j].features.empty());
    }
  }

  // Checks number of data.
  unsigned n = expected.size();
  while (reader.read(&sent)) ++n;
  BOOST_CHECK_EQUAL(500, n);
}

BOOST_AUTO_TEST_CASE(CheckReading_Ja) {
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

  SimpleTextReader reader("data/small.ja.tok");
  Sentence sent;

  // Checks read data.
  for (unsigned i = 0; i < expected.size(); ++i) {
    BOOST_CHECK(reader.read(&sent));
    BOOST_CHECK(sent.features.empty());
    BOOST_CHECK_EQUAL(expected[i].size(), sent.tokens.size());
    for (unsigned j = 0; j < expected[i].size(); ++j) {
      BOOST_CHECK_EQUAL(expected[i][j], sent.tokens[j].surface);
      BOOST_CHECK(sent.tokens[j].features.empty());
    }
  }

  // Checks number of data.
  unsigned n = expected.size();
  while (reader.read(&sent)) ++n;
  BOOST_CHECK_EQUAL(500, n);
}

BOOST_AUTO_TEST_SUITE_END()


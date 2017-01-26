#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>

#include <config.h>
#include <nmtkit/unicode.h>
#include <string>
#include <vector>

using std::string;
using std::vector;
using nmtkit::UTF8;

BOOST_AUTO_TEST_SUITE(UnicodeTest)

BOOST_AUTO_TEST_CASE(CheckUTF8_isFirstChar) {
  // true if c in {0x00..0x7f} or {0xc2..0xfd}
  const bool expected[256] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // 0x0*
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // 0x1*
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // 0x2*
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // 0x3*
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // 0x4*
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // 0x5*
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // 0x6*
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // 0x7*
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // 0x8*
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // 0x9*
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // 0xa*
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // 0xb*
    0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // 0xc*
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // 0xd*
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // 0xe*
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,  // 0xf*
  };

  for (unsigned i = 0; i < 256; ++i) {
    BOOST_CHECK_EQUAL(expected[i], UTF8::isFirstByte(i));
  }
}

BOOST_AUTO_TEST_CASE(CheckUTF8_getNumBytes) {
  // true if c in {0x00..0x7f} or {0xc2..0xfd}
  const unsigned expected[256] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // 0x0*
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // 0x1*
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // 0x2*
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // 0x3*
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // 0x4*
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // 0x5*
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // 0x6*
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // 0x7*
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // 0x8*
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // 0x9*
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // 0xa*
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // 0xb*
    0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  // 0xc*
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  // 0xd*
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,  // 0xe*
    4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 0, 0,  // 0xf*
  };

  for (unsigned i = 0; i < 256; ++i) {
    if (UTF8::isFirstByte(i)) {
      BOOST_CHECK_EQUAL(expected[i], UTF8::getNumBytes(i));
    }
  }
  BOOST_CHECK_EQUAL(1, UTF8::getNumBytes("0"[0]));
  BOOST_CHECK_EQUAL(1, UTF8::getNumBytes("A"[0]));
  BOOST_CHECK_EQUAL(2, UTF8::getNumBytes("Ä"[0]));
  BOOST_CHECK_EQUAL(2, UTF8::getNumBytes("א"[0]));
  BOOST_CHECK_EQUAL(3, UTF8::getNumBytes("ก"[0]));
  BOOST_CHECK_EQUAL(3, UTF8::getNumBytes("∀"[0]));
  BOOST_CHECK_EQUAL(3, UTF8::getNumBytes("あ"[0]));
  BOOST_CHECK_EQUAL(3, UTF8::getNumBytes("亜"[0]));
  BOOST_CHECK_EQUAL(4, UTF8::getNumBytes("𐎀"[0]));
  BOOST_CHECK_EQUAL(4, UTF8::getNumBytes("𐤀"[0]));
  BOOST_CHECK_EQUAL(4, UTF8::getNumBytes("🀀"[0]));
  BOOST_CHECK_EQUAL(4, UTF8::getNumBytes("🙂"[0]));
}

BOOST_AUTO_TEST_CASE(CheckUTF8_getLetters) {
  const vector<string> inputs {
    "This is a pen.",
    "これはテストです。",
    "🙂🙂🙂",
  };
  const vector<vector<string>> expected {
    {"T", "h", "i", "s", " ", "i", "s", " ", "a", " ", "p", "e", "n", "."},
    {"こ", "れ", "は", "テ", "ス", "ト", "で", "す", "。"},
    {"🙂", "🙂", "🙂"},
  };

  for (unsigned i = 0; i < inputs.size(); ++i) {
    const vector<string> observed = UTF8::getLetters(inputs[i]);
    BOOST_CHECK_EQUAL_COLLECTIONS(
        expected[i].begin(), expected[i].end(),
        observed.begin(), observed.end());
  }
}

BOOST_AUTO_TEST_SUITE_END()

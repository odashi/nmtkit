#include "config.h"

#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>

#include <vector>
#include <nmtkit/frequency_code.h>
#include <nmtkit/word_vocabulary.h>

using namespace std;

namespace {

const string tok_filename = "data/small.en.tok";
const string vocab_filename = "data/small.en.vocab";

template <class T>
void loadArchive(const string & filepath, T * obj) {
  ifstream ifs(filepath);
  boost::archive::binary_iarchive iar(ifs);
  iar >> *obj;
}

}  // namespace

BOOST_AUTO_TEST_SUITE(FrequencyCodeTest)


BOOST_AUTO_TEST_CASE(CheckNumBits) {
  const vector<unsigned> vocab_sizes {4, 7, 8, 9, 15, 16, 17};
  const vector<unsigned> num_bits {2, 3, 3, 4, 4, 4, 5};

  for (unsigned i = 0; i < vocab_sizes.size(); ++i) {
    nmtkit::WordVocabulary vocab(tok_filename, vocab_sizes[i]);
    nmtkit::FrequencyCode codec(vocab);
    BOOST_CHECK_EQUAL(num_bits[i], codec.getNumBits());
  }
}

BOOST_AUTO_TEST_CASE(CheckNumBits2) {
  nmtkit::WordVocabulary vocab;
  ::loadArchive(::vocab_filename, &vocab);
  nmtkit::FrequencyCode codec(vocab);
  BOOST_CHECK_EQUAL(9, codec.getNumBits());
}

BOOST_AUTO_TEST_CASE(CheckEncoding) {
  const vector<unsigned> ids {0, 1, 2, 3, 4, 10, 50, 100, 200, 400, 499};
  const vector<vector<bool>> expected {
    {0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 1, 0, 0, 0, 0, 0, 0, 0},
    {1, 1, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 1, 0, 0, 0, 0, 0, 0},
    {0, 1, 0, 1, 0, 0, 0, 0, 0},
    {0, 1, 0, 0, 1, 1, 0, 0, 0},
    {0, 0, 1, 0, 0, 1, 1, 0, 0},
    {0, 0, 0, 1, 0, 0, 1, 1, 0},
    {0, 0, 0, 0, 1, 0, 0, 1, 1},
    {1, 1, 0, 0, 1, 1, 1, 1, 1},
  };

  nmtkit::WordVocabulary vocab;
  ::loadArchive(::vocab_filename, &vocab);
  nmtkit::FrequencyCode codec(vocab);

  for (unsigned i = 0; i < ids.size(); ++i) {
    const vector<bool> observed = codec.getCode(ids[i]);
    BOOST_CHECK_EQUAL_COLLECTIONS(
        expected[i].begin(), expected[i].end(),
        observed.begin(), observed.end());
  }
}

BOOST_AUTO_TEST_CASE(CheckDecoding) {
  const vector<vector<bool>> code {
    {0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 1, 0, 0, 0, 0, 0, 0, 0},
    {1, 1, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 1, 0, 0, 0, 0, 0, 0},
    {0, 1, 0, 1, 0, 0, 0, 0, 0},
    {0, 1, 0, 0, 1, 1, 0, 0, 0},
    {0, 0, 1, 0, 0, 1, 1, 0, 0},
    {0, 0, 0, 1, 0, 0, 1, 1, 0},
    {0, 0, 0, 0, 1, 0, 0, 1, 1},
    {1, 1, 0, 0, 1, 1, 1, 1, 1},
    {0, 0, 1, 0, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1},
  };
  const vector<unsigned> expected {
    0, 1, 2, 3, 4, 10, 50, 100, 200, 400, 499,
    nmtkit::BinaryCode::INVALID_CODE,
    nmtkit::BinaryCode::INVALID_CODE,
  };

  nmtkit::WordVocabulary vocab;
  ::loadArchive(::vocab_filename, &vocab);
  nmtkit::FrequencyCode codec(vocab);

  for (unsigned i = 0; i < code.size(); ++i) {
    const unsigned observed = codec.getID(code[i]);
    BOOST_CHECK_EQUAL(expected[i], observed);
  }
}

BOOST_AUTO_TEST_SUITE_END()

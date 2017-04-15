#include "config.h"

#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>

#include <vector>
#include <nmtkit/identity_ecc.h>

using std::vector;
using nmtkit::IdentityECC;

BOOST_AUTO_TEST_SUITE(IdentityECCTest)

BOOST_AUTO_TEST_CASE(CheckNumBits) {
  IdentityECC ecc;
  for (unsigned i = 0; i < 1024; ++i) {
    BOOST_CHECK_EQUAL(i, ecc.getNumBits(i));
  }
}

BOOST_AUTO_TEST_CASE(CheckEncoding) {
  const vector<vector<bool>> inputs {
    {0, 0, 0, 0, 0, 0, 0, 0},
    {1, 1, 1, 1, 1, 1, 1, 1},
    {0, 1, 0, 1, 0, 1, 0, 1},
    {0, 0, 1, 1, 0, 0, 1, 1},
    {0, 0, 0, 0, 1, 1, 1, 1},
  };

  IdentityECC ecc;
  for (unsigned i = 0; i < inputs.size(); ++i) {
    const vector<bool> observed = ecc.encode(inputs[i]);
    BOOST_CHECK_EQUAL_COLLECTIONS(
        inputs[i].begin(), inputs[i].end(),
        observed.begin(), observed.end());
  }
}

BOOST_AUTO_TEST_CASE(CheckDecoding) {
  const vector<vector<float>> inputs {
    {0, 0, 0, 0, 0, 0, 0, 0},
    {1, 1, 1, 1, 1, 1, 1, 1},
    {0, 1, 0, 1, 0, 1, 0, 1},
    {0, 0, 1, 1, 0, 0, 1, 1},
    {0, 0, 0, 0, 1, 1, 1, 1},
  };

  IdentityECC ecc;
  for (unsigned i = 0; i < inputs.size(); ++i) {
    const vector<float> observed = ecc.decode(inputs[i]);
    BOOST_CHECK_EQUAL_COLLECTIONS(
        inputs[i].begin(), inputs[i].end(),
        observed.begin(), observed.end());
  }
}

BOOST_AUTO_TEST_SUITE_END()

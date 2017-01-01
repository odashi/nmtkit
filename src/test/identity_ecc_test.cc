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

BOOST_AUTO_TEST_SUITE_END()

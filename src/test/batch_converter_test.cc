#include "config.h"

#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>

#include <nmtkit/batch_converter.h>
#include <nmtkit/vocabulary.h>

using namespace std;

BOOST_AUTO_TEST_SUITE(BatchConverterTest)

BOOST_AUTO_TEST_CASE(CheckConvertion) {
  vector<nmtkit::Sample> input {
    { {10, 20, 30},         {10, 20, 30} },
    { {10, 20, 30, 40, 50}, {10, 20, 30, 40, 50} },
    { {10, 20},             {10, 20, 30, 40, 50} },
    { {10, 20, 30, 40, 50}, {10, 20, 30, 40} },
  };
  nmtkit::Batch expected {
    { { 1,  1,  1,  1},
      {10, 10, 10, 10},
      {20, 20, 20, 20},
      {30, 30,  2, 30},
      { 2, 40,  2, 40},
      { 2, 50,  2, 50},
      { 2,  2,  2,  2} },
    { { 1,  1,  1,  1},
      {10, 10, 10, 10},
      {20, 20, 20, 20},
      {30, 30, 30, 30},
      { 2, 40, 40, 40},
      { 2, 50, 50,  2},
      { 2,  2,  2,  2} },
  };
  nmtkit::Vocabulary src_vocab("data/small.en.vocab");
  nmtkit::Vocabulary trg_vocab("data/small.ja.vocab");
  nmtkit::BatchConverter conv(src_vocab, trg_vocab);
  nmtkit::Batch observed;
  conv.convert(input, &observed);
  for (unsigned i = 0; i < expected.source_ids.size(); ++i) {
    BOOST_CHECK_EQUAL_COLLECTIONS(
        expected.source_ids[i].begin(), expected.source_ids[i].end(),
        observed.source_ids[i].begin(), observed.source_ids[i].end());
  }
  for (unsigned i = 0; i < expected.target_ids.size(); ++i) {
    BOOST_CHECK_EQUAL_COLLECTIONS(
        expected.target_ids[i].begin(), expected.target_ids[i].end(),
        observed.target_ids[i].begin(), observed.target_ids[i].end());
  }
}

BOOST_AUTO_TEST_SUITE_END()

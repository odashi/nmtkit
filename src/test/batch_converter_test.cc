#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>

#include <nmtkit/batch_converter.h>

using namespace std;

BOOST_AUTO_TEST_SUITE(BatchConverterTest)

BOOST_AUTO_TEST_CASE(CheckConvertion) {
  vector<nmtkit::Sample> input {
    { {1, 2, 3},       {1, 2, 3} },
    { {1, 2, 3, 4, 5}, {1, 2, 3, 4, 5} },
    { {1, 2},          {1, 2, 3, 4, 5} },
    { {1, 2, 3, 4, 5}, {1, 2, 3, 4} },
  };
  nmtkit::Batch expected {
    { {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 0, 3}, {0, 4, 0, 4}, {0, 5, 0, 5} },
    { {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {0, 4, 4, 4}, {0, 5, 5, 0} },
  };

  nmtkit::Batch observed;
  nmtkit::BatchConverter::convert(input, 0, &observed);
  for (unsigned i = 0; i < expected.source_id.size(); ++i) {
    BOOST_CHECK_EQUAL_COLLECTIONS(
        expected.source_id[i].begin(), expected.source_id[i].end(),
        observed.source_id[i].begin(), observed.source_id[i].end());
  }
  for (unsigned i = 0; i < expected.target_id.size(); ++i) {
    BOOST_CHECK_EQUAL_COLLECTIONS(
        expected.target_id[i].begin(), expected.target_id[i].end(),
        observed.target_id[i].begin(), observed.target_id[i].end());
  }
}

BOOST_AUTO_TEST_SUITE_END()


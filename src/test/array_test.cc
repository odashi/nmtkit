#include "config.h"

#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>

#include <numeric>
#include <string>
#include <vector>
#include <nmtkit/array.h>

using std::greater;
using std::less;
using std::string;
using std::vector;

BOOST_AUTO_TEST_SUITE(ArrayTest)

BOOST_AUTO_TEST_CASE(CheckReversing) {
  const vector<vector<int>> test_data {
    {},
    {0},
    {0, 1},
    {0, 1, 2, 3, 4},
  };
  const vector<vector<int>> expected {
    {},
    {0},
    {1, 0},
    {4, 3, 2, 1, 0,}
  };

  for (unsigned i = 0; i < test_data.size(); ++i) {
    vector<int> result = test_data[i];
    nmtkit::Array::reverse(&result);
    BOOST_CHECK_EQUAL_COLLECTIONS(
        expected[i].begin(), expected[i].end(),
        result.begin(), result.end());
  }
}

BOOST_AUTO_TEST_CASE(CheckSortingIntegers) {
  const vector<vector<int>> test_data {
    {},
    {1},
    {1, 2, 3, 4, 5},
    {2, 4, 3, 1, 5},
    {5, 4, 3, 2, 1},
    {1, 2, 1, 3, 1, 2, 1},
  };
  const vector<vector<int>> expected_less {
    {},
    {1},
    {1, 2, 3, 4, 5},
    {1, 2, 3, 4, 5},
    {1, 2, 3, 4, 5},
    {1, 1, 1, 1, 2, 2, 3},
  };
  const vector<vector<int>> expected_greater {
    {},
    {1},
    {5, 4, 3, 2, 1},
    {5, 4, 3, 2, 1},
    {5, 4, 3, 2, 1},
    {3, 2, 2, 1, 1, 1, 1},
  };

  for (unsigned i = 0; i < test_data.size(); ++i) {
    vector<int> input = test_data[i];
    nmtkit::Array::sort(&input);  // less<int>()
    BOOST_CHECK_EQUAL_COLLECTIONS(
        expected_less[i].begin(), expected_less[i].end(),
        input.begin(), input.end());
    input = test_data[i];
    nmtkit::Array::sort(&input, greater<int>());
    BOOST_CHECK_EQUAL_COLLECTIONS(
        expected_greater[i].begin(), expected_greater[i].end(),
        input.begin(), input.end());
  }
}

BOOST_AUTO_TEST_CASE(CheckSortingVectors) {
  const vector<vector<string>> test_data {
    {"This", "is", "a", "test", "data", "."},
    {"This", "is", "an", "additional", "test", "data", "."},
    {"This", "is", "also", "a", "test", "data", "."},
    {"There are some another inputs.", "For example, this."},
    {"This is", "the last example."},
  };
  const vector<vector<string>> expected {
    {"This is", "the last example."},
    {"There are some another inputs.", "For example, this."},
    {"This", "is", "a", "test", "data", "."},
    {"This", "is", "also", "a", "test", "data", "."},
    {"This", "is", "an", "additional", "test", "data", "."},
  };

  vector<vector<string>> input = test_data;
  nmtkit::Array::sort(
      &input,
      [](const vector<string> & a, const vector<string> & b) {
          return a.size() < b.size();
      });
  for (unsigned i = 0; i < test_data.size(); ++i) {
    BOOST_CHECK_EQUAL_COLLECTIONS(
        expected[i].begin(), expected[i].end(),
        input[i].begin(), input[i].end());
  }
}

BOOST_AUTO_TEST_CASE(CheckShufflingPermutations) {
  nmtkit::Random rnd;
  const vector<unsigned> lengths {1, 2, 4, 8};
  const vector<unsigned> seeds {1, 10, 100};
  const vector<vector<vector<unsigned>>> expected_list {
    {{0}, {0}, {0}},
    {{0,1}, {1,0}, {1,0}},
    {{1,3,0,2}, {3,1,2,0}, {2,3,0,1}},
    {{3,7,6,1,4,5,2,0}, {6,3,2,5,0,4,7,1}, {4,5,3,1,2,6,7,0}},
  };

  for (unsigned l = 0; l < lengths.size(); ++l) {
    for (unsigned s = 0; s < seeds.size(); ++s) {
      vector<unsigned> samples(lengths[l]);
      iota(samples.begin(), samples.end(), 0);
      rnd.reset(seeds[s]);
      nmtkit::Array::shuffle(&samples, &rnd);
      BOOST_CHECK_EQUAL_COLLECTIONS(
          expected_list[l][s].begin(), expected_list[l][s].end(),
          samples.begin(), samples.end());
    }
  }
}

BOOST_AUTO_TEST_CASE(CheckShufflingDistributions) {
  nmtkit::Random rnd;
  const unsigned N = 100000;
  const vector<unsigned> lengths {1, 2, 4, 8};
  const vector<unsigned> seeds {1, 10, 100};

  for (const unsigned l : lengths) {
    for (const unsigned s : seeds) {
      vector<unsigned> samples(l);
      iota(samples.begin(), samples.end(), 0);
      vector<vector<unsigned>> freq(l, vector<unsigned>(l));
      rnd.reset(s);
      for (unsigned i = 0; i < N; ++i) {
        for (unsigned j = 0; j < l; ++j) {
          nmtkit::Array::shuffle(&samples, &rnd);
          for (unsigned k = 0; k < l; ++k) {
            ++freq[k][samples[k]];
          }
        }
      }
      for (unsigned i = 0; i < l; ++i) {
        for (unsigned j = 0; j < l; ++j) {
          BOOST_CHECK_CLOSE(static_cast<double>(N), freq[i][j], 1.0);
        }
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(CheckArgmax) {
  const vector<int> values {-2, -1, 0, 1, 2, 3};
  BOOST_CHECK_EQUAL(5, nmtkit::Array::argmax(values));  // greater<int>()
  BOOST_CHECK_EQUAL(0, nmtkit::Array::argmax(values, less<int>()));
  BOOST_CHECK_EQUAL(
      2,
      nmtkit::Array::argmax(values, [](int a, int b) { return -a*a > -b*b; }));
}

BOOST_AUTO_TEST_CASE(CheckKBest) {
  const vector<vector<int>> values {
    {0},
    {-1, 1},
    {0, -1, 2, 1, -2},
  };
  const vector<vector<vector<unsigned>>> expected_greater {
    { {0} },
    { {1}, {1, 0} },
    { {2}, {2, 3}, {2, 3, 0}, {2, 3, 0, 1}, {2, 3, 0, 1, 4} },
  };
  const vector<vector<vector<unsigned>>> expected_less {
    { {0} },
    { {0}, {0, 1} },
    { {4}, {4, 1}, {4, 1, 0}, {4, 1, 0, 3}, {4, 1, 0, 3, 2} },
  };
  const vector<vector<vector<unsigned>>> expected_sq {
    { {0} },
    { {1}, {1, 0} },
    { {4}, {4, 2}, {4, 2, 3}, {4, 2, 3, 1}, {4, 2, 3, 1, 0} },
  };

  for (unsigned i = 0; i < values.size(); ++i) {
    for (unsigned j = 0; j < values[i].size(); ++j) {
      vector<unsigned> results = nmtkit::Array::kbest(values[i], j + 1);
      BOOST_CHECK_EQUAL_COLLECTIONS(
          expected_greater[i][j].begin(), expected_greater[i][j].end(),
          results.begin(), results.end());

      results = nmtkit::Array::kbest(values[i], j + 1, less<int>());
      BOOST_CHECK_EQUAL_COLLECTIONS(
          expected_less[i][j].begin(), expected_less[i][j].end(),
          results.begin(), results.end());

      results = nmtkit::Array::kbest(values[i], j + 1, [](int a, int b) {
          return a*a > b*b;
      });
      BOOST_CHECK_EQUAL_COLLECTIONS(
          expected_sq[i][j].begin(), expected_sq[i][j].end(),
          results.begin(), results.end());
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()

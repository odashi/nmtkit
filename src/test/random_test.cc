#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>

#include <numeric>
#include <vector>
#include <nmtkit/random.h>

using namespace std;

BOOST_AUTO_TEST_SUITE(RandomTest)

BOOST_AUTO_TEST_CASE(CheckReset) {
  NMTKit::Random rnd;
  const int M = 1000000;
  const vector<unsigned> seeds {0, 1, 10, 100};
  const vector<vector<int>> expected_list {
    {  97874,  185956,  430700},
    {-165769,  994818,  440973},
    { 542988, -402344, -958487},
    {  87054,  342613, -443136},
  };
  
  // Checks the default seed.
  for (const int expected : expected_list[0]) {
    BOOST_CHECK_EQUAL(expected, rnd.uniform(-M, M));
  }

  // Checks user seeds twice.
  for (unsigned phase = 0; phase < 2; ++phase) {
    for (unsigned i = 0; i < seeds.size(); ++i) {
      rnd.reset(seeds[i]);
      for (const int expected : expected_list[i]) {
        BOOST_CHECK_EQUAL(expected, rnd.uniform(-M, M));
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(CheckUniformDistribution) {
  NMTKit::Random rnd;
  const unsigned N = 100000;
  const vector<unsigned> ranges {1, 2, 4, 8};
  const vector<unsigned> seeds {0, 1, 10, 100};
  for (const unsigned r : ranges) {
    for (const unsigned s : seeds) {
      vector<unsigned> freq(r);
      rnd.reset(s);
      const unsigned M = N * r;
      for (unsigned i = 0; i < M; ++i) {
        ++freq[rnd.uniform(0, r)];
      }
      for (unsigned i = 0; i < r; ++i) {
        BOOST_CHECK_CLOSE(static_cast<double>(N), freq[i], 2.0);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(CheckShufflingPermutations) {
  NMTKit::Random rnd;
  const vector<unsigned> lengths {1, 2, 4, 8};
  const vector<unsigned> seeds {0, 1, 10, 100};
  const vector<vector<vector<unsigned>>> expected_list {
    {{0}, {0}, {0}, {0}},
    {{1, 0}, {0, 1}, {1, 0}, {1, 0}},
    {{2, 0, 3, 1}, {1, 3, 0, 2}, {3, 1, 2, 0}, {2, 3, 0, 1}},
    {{4, 5, 6, 7, 2, 3, 1, 0}, {3, 7, 6, 1, 4, 5, 2, 0},
     {6, 3, 2, 5, 0, 4, 7, 1}, {4, 5, 3, 1, 2, 6, 7, 0}},
  };

  for (unsigned l = 0; l < lengths.size(); ++l) {
    for (unsigned s = 0; s < seeds.size(); ++s) {
      vector<unsigned> samples(lengths[l]);
      iota(samples.begin(), samples.end(), 0);
      rnd.reset(seeds[s]);
      rnd.shuffle(&samples);
      BOOST_CHECK_EQUAL_COLLECTIONS(
          expected_list[l][s].begin(), expected_list[l][s].end(),
          samples.begin(), samples.end());
    }
  }
}

BOOST_AUTO_TEST_CASE(CheckShufflingDistributions) {
  NMTKit::Random rnd;
  const unsigned N = 100000;
  const vector<unsigned> lengths {1, 2, 4, 8};
  const vector<unsigned> seeds {0, 1, 10, 100};

  for (const unsigned l : lengths) {
    for (const unsigned s : seeds) {
      vector<unsigned> samples(l);
      iota(samples.begin(), samples.end(), 0);
      vector<vector<unsigned>> freq(l, vector<unsigned>(l));
      rnd.reset(s);
      for (unsigned i = 0; i < N; ++i) {
        for (unsigned j = 0; j < l; ++j) {
          rnd.shuffle(&samples);
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

BOOST_AUTO_TEST_SUITE_END()


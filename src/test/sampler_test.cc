#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>

#include <iostream>
#include <vector>
#include <nmtkit/sampler.h>

using namespace std;

BOOST_AUTO_TEST_SUITE(SamplerTest)

BOOST_AUTO_TEST_CASE(CheckMonotoneIteration) {
  NMTKit::Vocabulary src_vocab("data/small.en.vocab");
  NMTKit::Vocabulary trg_vocab("data/small.ja.vocab");
  NMTKit::MonotoneSampler sampler(
      "data/small.en.tok", "data/small.ja.tok", src_vocab, trg_vocab);
  vector<vector<int>> expected_src = {
    {6, 41, 17, 90, 106, 37, 0, 364, 3},
    {159, 0, 13, 130, 0, 101, 332, 3},
    {6, 75, 12, 4, 145, 0, 3},
    {0, 219, 228, 3},
  };
  vector<vector<int>> expected_trg = {
    {86, 13, 198, 6, 142, 30, 22, 18, 6, 4, 245, 38, 20, 46, 29, 3},
    {268, 9, 472, 13, 283, 6, 33, 15, 10, 411, 69, 88, 8, 3},
    {18, 4, 158, 435, 12, 19, 3},
    {0, 4, 0, 164, 6, 309, 20, 19, 3},
  };

  // Checks first n samples
  vector<NMTKit::Sample> samples;
  for (int i = 0; i < expected_src.size(); ++i) {
    BOOST_CHECK(sampler.hasSamples());
    sampler.getSamples(&samples);
    BOOST_CHECK_EQUAL(1, samples.size());
    BOOST_CHECK_EQUAL_COLLECTIONS(
        expected_src[i].begin(), expected_src[i].end(),
        samples[0].source.begin(), samples[0].source.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(
        expected_trg[i].begin(), expected_trg[i].end(),
        samples[0].target.begin(), samples[0].target.end());
  }

  // Checks remaining samples
  for (int i = expected_src.size(); i < 500; ++i) {
    BOOST_CHECK(sampler.hasSamples());
    sampler.getSamples(&samples);
    BOOST_CHECK_EQUAL(1, samples.size());
  }
  BOOST_CHECK(!sampler.hasSamples());

  // Checks rewinding
  sampler.reset();
  for (int i = 0; i < expected_src.size(); ++i) {
    BOOST_CHECK(sampler.hasSamples());
    sampler.getSamples(&samples);
    BOOST_CHECK_EQUAL(1, samples.size());
    BOOST_CHECK_EQUAL_COLLECTIONS(
        expected_src[i].begin(), expected_src[i].end(),
        samples[0].source.begin(), samples[0].source.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(
        expected_trg[i].begin(), expected_trg[i].end(),
        samples[0].target.begin(), samples[0].target.end());
  }
}

BOOST_AUTO_TEST_SUITE_END()


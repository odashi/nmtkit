#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>

#include <iostream>
#include <vector>
#include <nmtkit/monotone_sampler.h>

using namespace std;

namespace {

const string src_tok_filename = "data/small.en.tok";
const string trg_tok_filename = "data/small.ja.tok";
const string src_vocab_filename = "data/small.en.vocab";
const string trg_vocab_filename = "data/small.ja.vocab";
const unsigned corpus_size = 500;  // #samples in the sample corpus
const unsigned max_length = 100;
const float max_length_ratio = 3.0;
const unsigned batch_size = 64;
const unsigned tail_size = corpus_size % batch_size;

const vector<vector<unsigned>> expected_src {
  {6, 41, 17, 90, 106, 37, 0, 364, 3},
  {159, 0, 13, 130, 0, 101, 332, 3},
  {6, 75, 12, 4, 145, 0, 3},
  {0, 219, 228, 3},
};
const vector<vector<unsigned>> expected_trg {
  {86, 13, 198, 6, 142, 30, 22, 18, 6, 4, 245, 38, 20, 46, 29, 3},
  {268, 9, 472, 13, 283, 6, 33, 15, 10, 411, 69, 88, 8, 3},
  {18, 4, 158, 435, 12, 19, 3},
  {0, 4, 0, 164, 6, 309, 20, 19, 3},
};

}  // namespace

BOOST_AUTO_TEST_SUITE(MonotoneSamplerTest)

BOOST_AUTO_TEST_CASE(CheckIteration) {
  nmtkit::Vocabulary src_vocab(::src_vocab_filename);
  nmtkit::Vocabulary trg_vocab(::trg_vocab_filename);
  nmtkit::MonotoneSampler sampler(
      ::src_tok_filename, ::trg_tok_filename,
      src_vocab, trg_vocab, ::max_length, ::max_length_ratio, ::batch_size);

  BOOST_CHECK(sampler.hasSamples());

  vector<nmtkit::Sample> samples;

  // Checks head samples.
  sampler.getSamples(&samples);
  BOOST_CHECK_EQUAL(::batch_size, samples.size());
  for (unsigned i = 0; i < ::expected_src.size(); ++i) {
    BOOST_CHECK_EQUAL_COLLECTIONS(
        ::expected_src[i].begin(), ::expected_src[i].end(),
        samples[i].source.begin(), samples[i].source.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(
        ::expected_trg[i].begin(), ::expected_trg[i].end(),
        samples[i].target.begin(), samples[i].target.end());
  }

  // Checks all iterations.
  while (sampler.hasSamples()) {
    sampler.getSamples(&samples);
    if (samples.size() != ::batch_size) {
      BOOST_CHECK_EQUAL(::tail_size, samples.size());
      BOOST_CHECK(!sampler.hasSamples());
    }
  }

  // Checks rewinding.
  sampler.rewind();
  BOOST_CHECK(sampler.hasSamples());

  // Re-checks head samples.
  sampler.getSamples(&samples);
  BOOST_CHECK_EQUAL(::batch_size, samples.size());
  for (unsigned i = 0; i < ::expected_src.size(); ++i) {
    BOOST_CHECK_EQUAL_COLLECTIONS(
        ::expected_src[i].begin(), ::expected_src[i].end(),
        samples[i].source.begin(), samples[i].source.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(
        ::expected_trg[i].begin(), ::expected_trg[i].end(),
        samples[i].target.begin(), samples[i].target.end());
  }
}

BOOST_AUTO_TEST_SUITE_END()

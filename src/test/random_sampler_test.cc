#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>

#include <iostream>
#include <vector>
#include <nmtkit/random_sampler.h>

using namespace std;

namespace {

const string src_tok_filename = "data/small.en.tok";
const string trg_tok_filename = "data/small.ja.tok";
const string src_vocab_filename = "data/small.en.vocab";
const string trg_vocab_filename = "data/small.ja.vocab";
const int corpus_size = 500;  // #samples in the sample corpus
const int seed = 12345;

const vector<vector<int>> expected_src {
  {18, 0, 9, 20, 16, 37, 33, 0, 142, 3},
  {35, 207, 35, 14, 451, 31, 14, 125, 3},
  {8, 19, 93, 5, 26, 198, 3},
  {4, 0, 312, 49, 4, 140, 3},
};
const vector<vector<int>> expected_trg {
  {0, 4, 33, 5, 489, 23, 25, 61, 5, 20, 19, 3},
  {21, 4, 100, 17, 0, 6, 145, 16, 8, 3},
  {14, 4, 21, 9, 237, 25, 97, 16, 8, 3},
  {463, 4, 133, 7, 216, 483, 15, 8, 3},
};
const vector<vector<int>> expected_src2 {
  {22, 195, 0, 5, 33, 295, 3},
  {208, 0, 399, 490, 348, 15, 400, 22, 103, 3},
  {7, 13, 5, 94, 0, 123, 7, 28, 5, 481, 3},
  {91, 14, 0, 31, 21, 41, 300, 3},
};
const vector<vector<int>> expected_trg2 {
  {14, 9, 166, 4, 0, 6, 429, 17, 3},
  {326, 7, 45, 99, 88, 17, 9, 13, 96, 11, 55, 4, 5, 11, 5, 3},
  {177, 31, 163, 347, 256, 16, 11, 104, 68, 11, 28, 11, 5, 3},
  {0, 21, 13, 476, 0, 68, 18, 51, 4, 291, 49, 20, 19, 3},
};

}  // namespace

BOOST_AUTO_TEST_SUITE(RandomSamplerTest)

BOOST_AUTO_TEST_CASE(CheckFiniteRandomIteration) {
  // NOTE: corpus_size should not be divisible by batch_size
  //       to check halfway samples.
  const int batch_size = 64;
  const int num_full_batches = ::corpus_size / batch_size;
  const int remain_size = ::corpus_size % batch_size;
  BOOST_REQUIRE(remain_size > 0);
  BOOST_REQUIRE(batch_size - remain_size >= ::expected_src.size());

  NMTKit::Vocabulary src_vocab(::src_vocab_filename);
  NMTKit::Vocabulary trg_vocab(::trg_vocab_filename);

  NMTKit::RandomSampler sampler(
      ::src_tok_filename, ::trg_tok_filename,
      src_vocab, trg_vocab, batch_size, false, ::seed);

  BOOST_CHECK_EQUAL(0, sampler.numIterated());
  BOOST_CHECK(sampler.hasSamples());

  vector<NMTKit::Sample> samples;

  // Checks first n samples.
  sampler.getSamples(&samples);
  BOOST_CHECK_EQUAL(batch_size, samples.size());
  BOOST_CHECK_EQUAL(batch_size, sampler.numIterated());
  for (int i = 0; i < ::expected_src.size(); ++i) {
    BOOST_CHECK_EQUAL_COLLECTIONS(
        ::expected_src[i].begin(), ::expected_src[i].end(),
        samples[i].source.begin(), samples[i].source.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(
        ::expected_trg[i].begin(), ::expected_trg[i].end(),
        samples[i].target.begin(), samples[i].target.end());
  }

  // Checks main iteration.
  for (int i = 1; i < num_full_batches; ++i) {
    BOOST_CHECK(sampler.hasSamples());
    sampler.getSamples(&samples);
    BOOST_CHECK_EQUAL(batch_size, samples.size());
    BOOST_CHECK_EQUAL((i + 1) * batch_size, sampler.numIterated());
  }

  // Checks remaining samples.
  BOOST_CHECK(sampler.hasSamples());
  sampler.getSamples(&samples);
  BOOST_CHECK_EQUAL(remain_size, samples.size());
  BOOST_CHECK_EQUAL(::corpus_size, sampler.numIterated());

  BOOST_CHECK(!sampler.hasSamples());

  // Checks resetting.
  sampler.reset();
  BOOST_CHECK_EQUAL(0, sampler.numIterated());
  BOOST_CHECK(sampler.hasSamples());

  // Re-checks first n samples.
  sampler.getSamples(&samples);
  BOOST_CHECK_EQUAL(batch_size, samples.size());
  BOOST_CHECK_EQUAL(batch_size, sampler.numIterated());
  for (int i = 0; i < ::expected_src.size(); ++i) {
    BOOST_CHECK_EQUAL_COLLECTIONS(
        ::expected_src[i].begin(), ::expected_src[i].end(),
        samples[i].source.begin(), samples[i].source.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(
        ::expected_trg[i].begin(), ::expected_trg[i].end(),
        samples[i].target.begin(), samples[i].target.end());
  }
}

BOOST_AUTO_TEST_CASE(CheckInfiniteRandomIteration) {
// NOTE: corpus_size should not be divisible by batch_size
//       to check halfway samples.
  const int batch_size = 64;
  const int num_full_batches = ::corpus_size / batch_size;
  const int remain_size = ::corpus_size % batch_size;
  BOOST_REQUIRE(remain_size > 0);
  BOOST_REQUIRE(batch_size - remain_size >= ::expected_src.size());

  NMTKit::Vocabulary src_vocab(::src_vocab_filename);
  NMTKit::Vocabulary trg_vocab(::trg_vocab_filename);

  NMTKit::RandomSampler sampler(
      ::src_tok_filename, ::trg_tok_filename,
      src_vocab, trg_vocab, batch_size, true, ::seed);

  BOOST_CHECK_EQUAL(0, sampler.numIterated());
  BOOST_CHECK(sampler.hasSamples());

  vector<NMTKit::Sample> samples;

  // Checks first n samples.
  sampler.getSamples(&samples);
  BOOST_CHECK_EQUAL(batch_size, samples.size());
  BOOST_CHECK_EQUAL(batch_size, sampler.numIterated());
  for (int i = 0; i < ::expected_src.size(); ++i) {
    BOOST_CHECK_EQUAL_COLLECTIONS(
        ::expected_src[i].begin(), ::expected_src[i].end(),
        samples[i].source.begin(), samples[i].source.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(
        ::expected_trg[i].begin(), ::expected_trg[i].end(),
        samples[i].target.begin(), samples[i].target.end());
  }

  // Checks main iteration.
  for (int i = 1; i < num_full_batches; ++i) {
    BOOST_CHECK(sampler.hasSamples());
    sampler.getSamples(&samples);
    BOOST_CHECK_EQUAL(batch_size, samples.size());
    BOOST_CHECK_EQUAL((i + 1) * batch_size, sampler.numIterated());
  }

  // Checks remaining/rewinding samples.
  BOOST_CHECK(sampler.hasSamples());
  sampler.getSamples(&samples);
  BOOST_CHECK_EQUAL(batch_size, samples.size());
  BOOST_CHECK_EQUAL(
      (num_full_batches + 1) * batch_size, sampler.numIterated());
  for (int i = 0; i < ::expected_src.size(); ++i) {
    BOOST_CHECK_EQUAL_COLLECTIONS(
        ::expected_src2[i].begin(), ::expected_src2[i].end(),
        samples[remain_size + i].source.begin(),
        samples[remain_size + i].source.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(
        ::expected_trg2[i].begin(), ::expected_trg2[i].end(),
        samples[remain_size + i].target.begin(),
        samples[remain_size + i].target.end());
  }

  BOOST_CHECK(sampler.hasSamples());
}

BOOST_AUTO_TEST_SUITE_END()


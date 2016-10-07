#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>

#include <iostream>
#include <vector>
#include <nmtkit/sampler.h>

using namespace std;

namespace {

const string src_tok_filename = "data/small.en.tok";
const string trg_tok_filename = "data/small.ja.tok";
const string src_vocab_filename = "data/small.en.vocab";
const string trg_vocab_filename = "data/small.ja.vocab";
const int corpus_size = 500;  // #samples in the sample corpus

const vector<vector<int>> expected_src = {
  {6, 41, 17, 90, 106, 37, 0, 364, 3},
  {159, 0, 13, 130, 0, 101, 332, 3},
  {6, 75, 12, 4, 145, 0, 3},
  {0, 219, 228, 3},
};
const vector<vector<int>> expected_trg = {
  {86, 13, 198, 6, 142, 30, 22, 18, 6, 4, 245, 38, 20, 46, 29, 3},
  {268, 9, 472, 13, 283, 6, 33, 15, 10, 411, 69, 88, 8, 3},
  {18, 4, 158, 435, 12, 19, 3},
  {0, 4, 0, 164, 6, 309, 20, 19, 3},
};

}  // namespace

BOOST_AUTO_TEST_SUITE(SamplerTest)

BOOST_AUTO_TEST_CASE(CheckFiniteMonotoneIteration) {
  // NOTE: corpus_size should not be divisible by batch_size
  //       to check halfway samples.
  const int batch_size = 64;
  const int num_full_batches = ::corpus_size / batch_size;
  const int remain_size = ::corpus_size % batch_size;
  BOOST_REQUIRE(remain_size > 0);
  BOOST_REQUIRE(batch_size - remain_size >= ::expected_src.size());

  NMTKit::Vocabulary src_vocab(::src_vocab_filename);
  NMTKit::Vocabulary trg_vocab(::trg_vocab_filename);
  NMTKit::Sampler sampler(
      ::src_tok_filename, ::trg_tok_filename,
      src_vocab, trg_vocab, batch_size, false);

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

BOOST_AUTO_TEST_CASE(CheckInfiniteMonotoneIteration) {
  // NOTE: corpus_size should not be divisible by batch_size
  //       to check halfway samples.
  const int batch_size = 64;
  const int num_full_batches = ::corpus_size / batch_size;
  const int remain_size = ::corpus_size % batch_size;
  BOOST_REQUIRE(remain_size > 0);
  BOOST_REQUIRE(batch_size - remain_size >= ::expected_src.size());

  NMTKit::Vocabulary src_vocab(::src_vocab_filename);
  NMTKit::Vocabulary trg_vocab(::trg_vocab_filename);
  NMTKit::Sampler sampler(
      ::src_tok_filename, ::trg_tok_filename,
      src_vocab, trg_vocab, batch_size, true);

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
        ::expected_src[i].begin(), ::expected_src[i].end(),
        samples[remain_size + i].source.begin(),
        samples[remain_size + i].source.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(
        ::expected_trg[i].begin(), ::expected_trg[i].end(),
        samples[remain_size + i].target.begin(),
        samples[remain_size + i].target.end());
  }

  BOOST_CHECK(sampler.hasSamples());
}

BOOST_AUTO_TEST_SUITE_END()


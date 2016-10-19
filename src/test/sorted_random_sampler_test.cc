#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>

#include <iostream>
#include <vector>
#include <nmtkit/sorted_random_sampler.h>

using namespace std;

namespace {

const string src_tok_filename = "data/small.en.tok";
const string trg_tok_filename = "data/small.ja.tok";
const string src_vocab_filename = "data/small.en.vocab";
const string trg_vocab_filename = "data/small.ja.vocab";
const unsigned corpus_size = 500;  // #samples in the sample corpus
const unsigned max_length = 100;
const unsigned num_words_in_batch = 256;
const unsigned random_seed = 12345;

const vector<vector<unsigned>> expected_src {
  {7, 77, 40, 39, 12, 0, 214, 270, 3},
  {22, 195, 0, 0, 3},
  {21, 95, 4, 395, 115, 0, 3},
  {22, 344, 44, 5, 24, 36, 465, 3},
};
const vector<vector<unsigned>> expected_trg {
  {485, 0, 137, 330, 6, 142, 30, 345, 12, 19, 3},
  {14, 9, 166, 4, 343, 7, 192, 20, 46, 8, 3},
  {18, 87, 4, 27, 179, 7, 0, 6, 16, 8, 3},
  {14, 9, 0, 4, 107, 25, 319, 13, 32, 17, 3},
};
const vector<vector<unsigned>> expected_src2 {
  {6, 75, 61, 5, 0, 7, 10, 264, 29, 414, 15, 0, 57, 34, 91, 3},
  {208, 0, 242, 5, 78, 274, 12, 23, 0, 273, 3},
  {6, 13, 5, 184, 10, 221, 3},
  {111, 5, 68, 8, 37, 40, 93, 5, 26, 3},
};
const vector<vector<unsigned>> expected_trg2 {
  {67, 139, 137, 9, 41, 0, 7, 0, 6, 484, 16, 10, 41, 30, 3},
  {86, 24, 361, 427, 11, 269, 6, 116, 7, 239, 10, 165, 11, 5, 3},
  {109, 22, 210, 68, 11, 28, 11, 5, 110, 13, 32, 38, 20, 19, 3},
  {0, 0, 34, 0, 63, 14, 4, 21, 25, 97, 31, 23, 81, 26, 3},
};

const vector<unsigned> expected_batch_sizes {
  21, 28, 21, 18, 18, 16, 32, 17, 19, 19,
  36, 23, 15, 25, 25, 17, 23, 18, 16, 21,
  28, 19, 25,
};
const vector<unsigned> expected_lengths {
  12,  9, 12, 14, 14, 16,  8, 15, 13, 13,
   7, 11, 16, 10, 10, 15, 11, 14, 16, 12,
   9, 13, 10,
};

}  // namespace

BOOST_AUTO_TEST_SUITE(SortedRandomSamplerTest)

BOOST_AUTO_TEST_CASE(CheckIteration) {
  // Prechecks test data.
  BOOST_REQUIRE_EQUAL(::expected_batch_sizes.size(), ::expected_lengths.size());
  unsigned total_num_samples = 0;
  for (unsigned i = 0; i < expected_batch_sizes.size(); ++i) {
    total_num_samples += expected_batch_sizes[i];
    BOOST_REQUIRE_LE(
        ::expected_batch_sizes[i] * ::expected_lengths[i],
        num_words_in_batch);
  }
  BOOST_REQUIRE_EQUAL(corpus_size, total_num_samples);

  nmtkit::Vocabulary src_vocab(::src_vocab_filename);
  nmtkit::Vocabulary trg_vocab(::trg_vocab_filename);
  nmtkit::SortedRandomSampler sampler(
      ::src_tok_filename, ::trg_tok_filename,
      src_vocab, trg_vocab, ::max_length, ::num_words_in_batch, ::random_seed);

  BOOST_CHECK(sampler.hasSamples());

  vector<nmtkit::Sample> samples;
  vector<unsigned> batch_sizes;
  vector<unsigned> lengths;

  // Checks head samples.
  sampler.getSamples(&samples);
  batch_sizes.emplace_back(samples.size());
  lengths.emplace_back(samples.back().target.size());
  BOOST_CHECK_LE(samples.front().target.size(), samples.back().target.size());

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
    batch_sizes.emplace_back(samples.size());
    lengths.emplace_back(samples.back().target.size());
    BOOST_CHECK_LE(samples.front().target.size(), samples.back().target.size());
  }
  BOOST_CHECK_EQUAL_COLLECTIONS(
      ::expected_batch_sizes.begin(), ::expected_batch_sizes.end(),
      batch_sizes.begin(), batch_sizes.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(
      ::expected_lengths.begin(), ::expected_lengths.end(),
      lengths.begin(), lengths.end());

  // Checks rewinding.
  sampler.rewind();
  BOOST_CHECK(sampler.hasSamples());

  // Re-checks head samples.
  // The order of samples was shuffled again by calling rewind(), and generated
  // batch has different samples with the first one.
  sampler.getSamples(&samples);
  for (unsigned i = 0; i < ::expected_src2.size(); ++i) {
    BOOST_CHECK_EQUAL_COLLECTIONS(
        ::expected_src2[i].begin(), ::expected_src2[i].end(),
        samples[i].source.begin(), samples[i].source.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(
        ::expected_trg2[i].begin(), ::expected_trg2[i].end(),
        samples[i].target.begin(), samples[i].target.end());
  }
}

BOOST_AUTO_TEST_SUITE_END()


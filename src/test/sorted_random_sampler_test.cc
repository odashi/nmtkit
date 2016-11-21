#include "config.h"

#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>

#include <fstream>
#include <vector>
#include <boost/archive/text_iarchive.hpp>
#include <nmtkit/sorted_random_sampler.h>
#include <nmtkit/word_vocabulary.h>

using namespace std;

namespace {

const string src_tok_filename = "data/small.en.tok";
const string trg_tok_filename = "data/small.ja.tok";
const string src_vocab_filename = "data/small.en.vocab";
const string trg_vocab_filename = "data/small.ja.vocab";
const unsigned corpus_size = 500;  // #samples in the sample corpus
const unsigned max_length = 100;
const float max_length_ratio = 3.0;
const unsigned num_words_in_batch = 256;
const unsigned random_seed = 12345;

const vector<vector<unsigned>> expected_src {
  { 21, 351,  65,  60,   0,  15, 193,   3},
  {143, 172,  17, 149,  35, 366,  35, 397,   3},
  { 63,  43,  12,  56,  94, 261,  34, 227,   3},
  { 21, 196,  13, 170,  59,   5,   0,  16,   3},
};
const vector<vector<unsigned>> expected_trg {
  {184,  31,  36,   4, 211, 273,  16,  10,  11,   5,   3},
  {157,   4, 205,   0, 237,  30, 442,  28,  11,   5,   3},
  {419,   6,  98,  15,  10,   0,   6, 100,  15,  10,   3},
  { 50,   7, 448,  31,  95,   4, 102, 302,  32,  17,   3},
};
const vector<vector<unsigned>> expected_src2 {
  {  6,  37,  33,   0,   5,  81,   7,   0,   3},
  {  6,  46,  17,   0,   7,  27, 237,  20,   3},
  {408,   0,   5,   4, 321,   5, 152, 145,   3},
  { 14, 194, 128,  35,  98,  35,  14,  83,   3},
};
const vector<vector<unsigned>> expected_trg2 {
  { 54,  12,  24, 149,  29,  12,  41, 284,   0,  16,  20,  19,   3},
  { 18,   4,  43,  13, 164,  16,   8,  36,   7,   0,  11,   5,   3},
  {  0,   4, 158,   7,  16,   6, 315,  64,  45,  37,  20,  19,   3},
  { 21,   4, 312,  17, 216, 119, 144, 135,   7,  97,  16,   8,   3},
};

const vector<unsigned> expected_batch_sizes {
  19, 25, 19, 17, 17, 15, 28, 16, 18, 18,
  21, 32, 11, 23, 18, 17, 23, 16, 19, 21,
  25, 23, 15, 25, 19,
};
const vector<unsigned> expected_lengths {
  11,  9, 12, 14, 14, 16,  8, 15, 13, 13,
  11,  7, 16, 10, 13, 14, 10, 15, 12, 11,
   9, 10, 16,  9, 12,
};

template <class T>
void loadArchive(const string & filepath, T * obj) {
  ifstream ifs(filepath);
  boost::archive::text_iarchive iar(ifs);
  iar >> *obj;
}

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

  nmtkit::WordVocabulary src_vocab, trg_vocab;
  ::loadArchive(::src_vocab_filename, &src_vocab);
  ::loadArchive(::trg_vocab_filename, &trg_vocab);
  nmtkit::SortedRandomSampler sampler(
      ::src_tok_filename, ::trg_tok_filename,
      src_vocab, trg_vocab,
      "target_word", "target_source",
      ::num_words_in_batch, ::max_length, ::max_length_ratio, ::random_seed);

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

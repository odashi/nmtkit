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
  {22, 195, 0, 0, 3},
  {21, 95, 4, 395, 115, 0, 3},
  {22, 344, 44, 5, 24, 36, 465, 3},
  {26, 191, 9, 10, 0, 376, 3},
};
const vector<vector<unsigned>> expected_trg {
  {14, 9, 173, 4, 345, 7, 181, 20, 46, 8, 3},
  {18, 85, 4, 27, 189, 7, 0, 6, 16, 8, 3},
  {14, 9, 481, 4, 106, 25, 231, 13, 32, 17, 3},
  {21, 9, 368, 127, 4, 0, 11, 303, 12, 19, 3},
};
const vector<vector<unsigned>> expected_src2 {
  {8, 175, 22, 336, 67, 3},
  {70, 29, 334, 255, 302, 3},
  {106, 9, 61, 5, 277, 23, 246, 11},
  {391, 30, 4, 240, 3},
};
const vector<vector<unsigned>> expected_trg2 {
  {14, 4, 78, 34, 41, 292, 7, 60, 15, 10, 76, 8, 3},
  {161, 436, 5, 369, 6, 0, 6, 129, 38, 20, 65, 26, 3},
  {86, 13, 39, 0, 7, 464, 30, 9, 12, 65, 26, 22, 3},
  {18, 6, 27, 171, 7, 60, 15, 10, 37, 11, 69, 5, 3},
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
      src_vocab, trg_vocab, ::max_length, ::max_length_ratio,
      ::num_words_in_batch, ::random_seed);

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

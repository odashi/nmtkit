#include "config.h"

#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>

#include <fstream>
#include <vector>
#include <boost/archive/text_iarchive.hpp>
#include <nmtkit/random_sampler.h>
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
const unsigned batch_size = 64;
const unsigned tail_size = corpus_size % batch_size;
const unsigned random_seed = 12345;

const vector<vector<unsigned>> expected_src {
  {18, 0, 9, 20, 16, 37, 33, 0, 142, 3},
  {35, 207, 35, 14, 451, 31, 14, 125, 3},
  {8, 19, 93, 5, 26, 198, 3},
  {4, 0, 312, 49, 4, 140, 3},
};
const vector<vector<unsigned>> expected_trg {
  {0, 4, 33, 5, 0, 23, 25, 61, 5, 20, 19, 3},
  {21, 4, 100, 17, 0, 6, 148, 16, 8, 3},
  {14, 4, 21, 9, 295, 25, 92, 16, 8, 3},
  {0, 4, 141, 7, 219, 0, 15, 8, 3},
};
const vector<vector<unsigned>> expected_src2 {
  {22, 195, 0, 5, 33, 295, 3},
  {208, 0, 399, 490, 348, 15, 400, 22, 103, 3},
  {7, 13, 5, 94, 0, 123, 7, 28, 5, 481, 3},
  {91, 14, 0, 31, 21, 41, 300, 3},
};
const vector<vector<unsigned>> expected_trg2 {
  {14, 9, 173, 4, 0, 6, 482, 17, 3},
  {228, 7, 44, 99, 88, 17, 9, 13, 96, 11, 56, 4, 5, 11, 5, 3},
  {184, 31, 163, 349, 304, 16, 11, 104, 68, 11, 28, 11, 5, 3},
  {0, 21, 13, 0, 0, 68, 18, 51, 4, 311, 49, 20, 19, 3},
};

template <class T>
void loadArchive(const string & filepath, T * obj) {
  ifstream ifs(filepath);
  boost::archive::text_iarchive iar(ifs);
  iar >> *obj;
}

}  // namespace

BOOST_AUTO_TEST_SUITE(RandomSamplerTest)

BOOST_AUTO_TEST_CASE(CheckIteration) {
  nmtkit::WordVocabulary src_vocab, trg_vocab;
  ::loadArchive(::src_vocab_filename, &src_vocab);
  ::loadArchive(::trg_vocab_filename, &trg_vocab);
  nmtkit::RandomSampler sampler(
      ::src_tok_filename, ::trg_tok_filename,
      src_vocab, trg_vocab, ::max_length, ::max_length_ratio,
      ::batch_size, ::random_seed);

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
  // The order of samples was shuffled again by calling rewind(), and generated
  // batch has different samples with the first one.
  sampler.getSamples(&samples);
  BOOST_CHECK_EQUAL(::batch_size, samples.size());
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

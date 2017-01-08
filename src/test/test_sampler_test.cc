#include "config.h"

#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>

#include <fstream>
#include <vector>
#include <boost/archive/text_iarchive.hpp>
#include <nmtkit/test_sampler.h>
#include <nmtkit/word_vocabulary.h>

using namespace std;

namespace {

const string src_tok_filename = "data/small.en.tok";
const string trg_tok_filename = "data/small.ja.tok";
const string src_vocab_filename = "data/small.en.vocab";
const string trg_vocab_filename = "data/small.ja.vocab";
const unsigned corpus_size = 500;  // #samples in the sample corpus
const unsigned batch_size = 64;
const unsigned tail_size = corpus_size % batch_size;

const vector<vector<unsigned>> expected_src {
  {6, 41, 17, 90, 106, 37, 0, 364, 3},
  {159, 0, 13, 130, 0, 101, 332, 3},
  {6, 75, 12, 4, 145, 0, 3},
  {0, 219, 228, 3},
};
const vector<string> expected_src_strings {
  "i can 't tell who will arrive first .",
  "many animals have been destroyed by men .",
  "i 'm in the tennis club .",
  "emi looks happy ."
};
const vector<vector<unsigned>> expected_trg {
  {86, 13, 202, 6, 138, 30, 22, 18, 6, 4, 310, 38, 20, 46, 29, 3},
  {298, 9, 0, 13, 325, 6, 33, 15, 10, 0, 69, 88, 8, 3},
  {18, 4, 158, 416, 12, 19, 3},
  {0, 4, 0, 164, 6, 242, 20, 19, 3},
};
const vector<string> expected_trg_strings {
  "誰 が 一番 に 着 く か 私 に は 分か り ま せ ん 。",
  "多く の 動物 が 人間 に よ っ て 滅ぼ さ れ た 。",
  "私 は テニス 部員 で す 。",
  "エミ は 幸せ そう に 見え ま す 。"
};

template <class T>
void loadArchive(const string & filepath, T * obj) {
  ifstream ifs(filepath);
  boost::archive::binary_iarchive iar(ifs);
  iar >> *obj;
}

}  // namespace

BOOST_AUTO_TEST_SUITE(TestSamplerTest)

BOOST_AUTO_TEST_CASE(CheckIteration) {
  nmtkit::WordVocabulary src_vocab, trg_vocab;
  ::loadArchive(::src_vocab_filename, &src_vocab);
  ::loadArchive(::trg_vocab_filename, &trg_vocab);
  nmtkit::TestSampler sampler(
      ::src_tok_filename, ::trg_tok_filename,
      src_vocab, trg_vocab, ::batch_size);

  BOOST_CHECK(sampler.hasSamples());

  // Checks head samples.
  {
    vector<nmtkit::Sample> samples = sampler.getSamples();
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

  // Checks rewinding.
  sampler.rewind();
  BOOST_CHECK(sampler.hasSamples());

  // Re-checks head samples.
  {
    vector<nmtkit::Sample> samples = sampler.getSamples();
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
}

BOOST_AUTO_TEST_CASE(CheckIteration2) {
  nmtkit::WordVocabulary src_vocab, trg_vocab;
  ::loadArchive(::src_vocab_filename, &src_vocab);
  ::loadArchive(::trg_vocab_filename, &trg_vocab);
  nmtkit::TestSampler sampler(
      ::src_tok_filename, ::trg_tok_filename,
      src_vocab, trg_vocab, ::batch_size);

  BOOST_CHECK(sampler.hasSamples());

  // Checks head samples.
  {
    vector<nmtkit::TestSample> samples = sampler.getTestSamples();
    BOOST_CHECK_EQUAL(::batch_size, samples.size());
    for (unsigned i = 0; i < ::expected_src.size(); ++i) {
      BOOST_CHECK_EQUAL_COLLECTIONS(
          ::expected_src[i].begin(), ::expected_src[i].end(),
          samples[i].source.begin(), samples[i].source.end());
      BOOST_CHECK_EQUAL(expected_src_strings[i], samples[i].source_string);
      BOOST_CHECK_EQUAL_COLLECTIONS(
          ::expected_trg[i].begin(), ::expected_trg[i].end(),
          samples[i].target.begin(), samples[i].target.end());
      BOOST_CHECK_EQUAL(expected_trg_strings[i], samples[i].target_string);
    }
  }

  // Checks rewinding.
  sampler.rewind();
  BOOST_CHECK(sampler.hasSamples());

  // Re-checks head samples.
  {
    vector<nmtkit::Sample> samples = sampler.getSamples();
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
}

BOOST_AUTO_TEST_SUITE_END()

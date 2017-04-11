#include <config.h>

#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>

#include <algorithm>
#include <fstream>
#include <vector>
#include <boost/archive/text_iarchive.hpp>
#include <nmtkit/sorted_random_sampler.h>
#include <nmtkit/word_vocabulary.h>

using std::ifstream;
using std::max;
using std::string;
using std::vector;

namespace globals {

const string src_tok_filename = "data/small.en.tok";
const string trg_tok_filename = "data/small.ja.tok";
const string src_vocab_filename = "data/small.en.vocab";
const string trg_vocab_filename = "data/small.ja.vocab";
const unsigned max_length = 16;
const float max_length_ratio = 3.0;
const unsigned random_seed = 12345;

template <class T>
void loadArchive(const string & filepath, T * obj) {
  ifstream ifs(filepath);
  boost::archive::binary_iarchive iar(ifs);
  iar >> *obj;
}

}  // namespace

BOOST_AUTO_TEST_SUITE(SortedRandomSamplerTest)

BOOST_AUTO_TEST_CASE(CheckRewinding) {
  const vector<vector<unsigned>> expected_src {
    {  6, 13,  5, 40, 64,119,  0,  3},
    { 21,351, 65, 60,  0, 15,193,  3},
    {143,172, 17,149, 35,366, 35,397,  3},
    { 63, 43, 12, 56, 94,261, 34,227,  3},
  };
  const vector<vector<unsigned>> expected_trg {
    {  0,114,  5,  0,  7, 91, 99, 11, 30,  0,  3},
    {184, 31, 36,  4,211,273, 16, 10, 11,  5,  3},
    {157,  4,205,  0,237, 30,442, 28, 11,  5,  3},
    {419,  6, 98, 15, 10,  0,  6,100, 15, 10,  3},
  };
  const vector<vector<unsigned>> expected_src2 {
    { 62,  8, 90,  7,  4,192, 11},
    {208,  0, 25, 37,357,209,  3},
    { 21, 28, 38,177, 27,  0,  3},
    {  8, 77, 13,475,  4,233,  3},
  };
  const vector<vector<unsigned>> expected_trg2 {
    { 14,  4, 42,  6,140,  9, 36,  7, 44,  5, 20, 16,  8, 22,  3},
    {271,  6, 35, 13, 90, 17, 22,215,  6, 24,120, 28, 11,  5,  3},
    {  0,  4, 74,216,  9, 83,  6,139,  8,  9, 12,  4, 11,  5,  3},
    { 14,  4,134,  7,  0, 17,122, 37, 12, 32, 15,  8,  9,  6,  3},
  };

  nmtkit::WordVocabulary src_vocab, trg_vocab;
  globals::loadArchive(globals::src_vocab_filename, &src_vocab);
  globals::loadArchive(globals::trg_vocab_filename, &trg_vocab);
  nmtkit::SortedRandomSampler sampler(
      globals::src_tok_filename, globals::trg_tok_filename,
      src_vocab, trg_vocab,
      "target_word", "target_source", 256,
      globals::max_length, globals::max_length_ratio, globals::random_seed);

  BOOST_CHECK(sampler.hasSamples());

  // Checks head samples.
  {
    vector<nmtkit::Sample> samples = sampler.getSamples();
    for (unsigned i = 0; i < expected_src.size(); ++i) {
      BOOST_CHECK_EQUAL_COLLECTIONS(
          expected_src[i].begin(), expected_src[i].end(),
          samples[i].source.begin(), samples[i].source.end());
      BOOST_CHECK_EQUAL_COLLECTIONS(
          expected_trg[i].begin(), expected_trg[i].end(),
          samples[i].target.begin(), samples[i].target.end());
    }
  }

  // Skips all iterations.
  while (sampler.hasSamples()) {
    sampler.getSamples();
  }

  // Checks rewinding.
  sampler.rewind();
  BOOST_CHECK(sampler.hasSamples());

  // Re-checks head samples.
  // The order of samples was shuffled again by calling rewind(), and generated
  // batch has different samples with the first one.
  {
    vector<nmtkit::Sample> samples = sampler.getSamples();
    for (unsigned i = 0; i < expected_src2.size(); ++i) {
      BOOST_CHECK_EQUAL_COLLECTIONS(
          expected_src2[i].begin(), expected_src2[i].end(),
          samples[i].source.begin(), samples[i].source.end());
      BOOST_CHECK_EQUAL_COLLECTIONS(
          expected_trg2[i].begin(), expected_trg2[i].end(),
          samples[i].target.begin(), samples[i].target.end());
    }
  }
}

BOOST_AUTO_TEST_CASE(CheckSorting) {
  const vector<string> sort_methods {
    "none",
    "source",
    "target",
    "source_target",
    "target_source",
  };
  const vector<vector<vector<unsigned>>> expected_src {
    // none
    {{ 12, 10,307, 31,162,  9,102, 10,  0,  3},
     {  4,126,  9,342,  5,369,  3},
     {224,  9,270, 12,  4,  0, 15,  4,299,  3},
     { 42,  0, 38,160, 30, 12,  4,367,  3}},
    // source
    {{433, 27, 32,448, 31, 50,  3},
     {  6,259,489, 49, 27,  7,  3},
     {107, 28,  7,  0,  5,146, 11},
     {  6,  0,  5, 13,168,  0,  3}},
    // target
    {{  7, 77, 40, 39, 12,  0,214,270,  3},
     { 22,195,  0,  0,  3},
     { 21, 95,  4,395,115,  0,  3},
     { 22,344, 44,  5, 24, 36,465,  3}},
    // source_target
    {{ 25, 24,  7, 24, 12,225, 11},
     {  4,166,  9,313,149,171,  3},
     {173, 14, 48, 12, 23,  0, 11},
     {  6, 13, 92,  5, 24,118,  3}},
    // target_source
    {{  6, 13,  5, 40, 64,119,  0,  3},
     { 21,351, 65, 60,  0, 15,193,  3},
     {143,172, 17,149, 35,366, 35,397,  3},
     { 63, 43, 12, 56, 94,261, 34,227,  3}},
  };
  const vector<vector<vector<unsigned>>> expected_trg {
    // none
    {{  0,  0, 12, 34,326,  4,  0,  6,126, 11,  5,  3},
     { 27,155,  4,360,  9, 56,  6, 75, 28, 88, 10,  5, 17,  3},
     {208,  4,223,  9,113, 64, 25,268,  5, 10,  5, 17,  3},
     { 14, 28,  4,  0,  6, 18,  9,130,  7,109,341, 11, 40,  8,  3}},
    // source
    {{394,  7, 41,188,357, 80,  5,  3},
     { 18,  4, 42,  7,  0, 16, 20, 19,  3,},
     {352,  6,180, 31,329, 12, 19, 22,  3,},
     { 18,  4,412,  6, 45, 30,163,103,  8,  9, 12, 19, 13,  3}},
    // target
    {{  0,377,151,323,  6,138, 30,347, 12, 19,  3},
     { 14,  9,173,  4,345,  7,181, 20, 46,  8,  3},
     { 18, 85,  4, 27,189,  7,  0,  6, 16,  8,  3},
     { 14,  9,481,  4,106, 25,231, 13, 32, 17,  3}},
    // source_target
    {{112, 12,  4,  0, 52,  7, 16, 10,  5, 20, 19, 22,  3},
     { 27,117,  4,367,  0,  0,  6, 11, 15, 10,  5, 17,  3},
     { 21,  4, 39,179, 12,  0,  5, 10,  5, 20, 19, 22,  3},
     { 18,  4,  0,  4,350, 24, 31, 36, 13, 49, 20, 46, 29,  3}},
    // target_source
    {{  0,114,  5,  0,  7, 91, 99, 11, 30,  0,  3},
     {184, 31, 36,  4,211,273, 16, 10, 11,  5,  3},
     {157,  4,205,  0,237, 30,442, 28, 11,  5,  3},
     {419,  6, 98, 15, 10,  0,  6,100, 15, 10,  3}},
  };

  nmtkit::WordVocabulary src_vocab, trg_vocab;
  globals::loadArchive(globals::src_vocab_filename, &src_vocab);
  globals::loadArchive(globals::trg_vocab_filename, &trg_vocab);

  for (unsigned i = 0; i < sort_methods.size(); ++i) {
    nmtkit::SortedRandomSampler sampler(
        globals::src_tok_filename, globals::trg_tok_filename,
        src_vocab, trg_vocab,
        "target_word", sort_methods[i], 256,
        globals::max_length, globals::max_length_ratio, globals::random_seed);

    BOOST_CHECK(sampler.hasSamples());

    // Checks only head samples.
    vector<nmtkit::Sample> samples = sampler.getSamples();
    for (unsigned j = 0; j < expected_src[i].size(); ++j) {
      BOOST_CHECK_EQUAL_COLLECTIONS(
          expected_src[i][j].begin(), expected_src[i][j].end(),
          samples[j].source.begin(), samples[j].source.end());
      BOOST_CHECK_EQUAL_COLLECTIONS(
          expected_trg[i][j].begin(), expected_trg[i][j].end(),
          samples[j].target.begin(), samples[j].target.end());
    }
  }
}

BOOST_AUTO_TEST_CASE(CheckBatch_Sentence) {
  const vector<unsigned> batch_sizes {
    1, 2, 3, 5, 7, 10, 100, 200, 300, 500, 700,
  };

  nmtkit::WordVocabulary src_vocab, trg_vocab;
  globals::loadArchive(globals::src_vocab_filename, &src_vocab);
  globals::loadArchive(globals::trg_vocab_filename, &trg_vocab);

  for (const unsigned batch_size : batch_sizes) {
    nmtkit::SortedRandomSampler sampler(
        globals::src_tok_filename, globals::trg_tok_filename,
        src_vocab, trg_vocab,
        "sentence", "none", batch_size,
        globals::max_length, globals::max_length_ratio, globals::random_seed);

    unsigned num_data = 0;
    bool tail_sample = false;

    while (sampler.hasSamples()) {
      vector<nmtkit::Sample> samples = sampler.getSamples();
      num_data += samples.size();
      if (samples.size() != batch_size) {
        BOOST_CHECK(!tail_sample);
        BOOST_CHECK_EQUAL(500 % batch_size, samples.size());
        tail_sample = true;
      }
    }
    BOOST_CHECK_EQUAL(500, num_data);
    BOOST_CHECK_EQUAL(500 % batch_size != 0, tail_sample);
  }
}

BOOST_AUTO_TEST_CASE(CheckBatch_BothWord) {
  const vector<unsigned> expected_batch_sizes {
    22,28,26,20,20,17,26,18,22,25,
    22,34, 4,21,23,24,25,28,32,19,
    23,21,
    // sum = 500
  };
  const vector<unsigned> expected_lengths {
    23,18,19,25,25,29,19,27,22,20,
    23,15,31,24,22,21,20,18,16,26,
    22,24,
  };

  nmtkit::WordVocabulary src_vocab, trg_vocab;
  globals::loadArchive(globals::src_vocab_filename, &src_vocab);
  globals::loadArchive(globals::trg_vocab_filename, &trg_vocab);

  nmtkit::SortedRandomSampler sampler(
      globals::src_tok_filename, globals::trg_tok_filename,
      src_vocab, trg_vocab,
      "both_word", "source_target", 512,
      globals::max_length, globals::max_length_ratio, globals::random_seed);

  vector<unsigned> batch_sizes;
  vector<unsigned> lengths;

  while (sampler.hasSamples()) {
    vector<nmtkit::Sample> samples = sampler.getSamples();
    batch_sizes.emplace_back(samples.size());
    unsigned max_length = 0;
    for (auto sample : samples) {
      const unsigned cur_length = sample.source.size() + sample.target.size();
      max_length = max(max_length, cur_length);
    }
    lengths.emplace_back(max_length);
  }
  BOOST_CHECK_EQUAL_COLLECTIONS(
      expected_batch_sizes.begin(), expected_batch_sizes.end(),
      batch_sizes.begin(), batch_sizes.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(
      expected_lengths.begin(), expected_lengths.end(),
      lengths.begin(), lengths.end());
}

BOOST_AUTO_TEST_CASE(CheckBatch_SourceWord) {
  const vector<unsigned> expected_batch_sizes {
    32,42,28,25,25,15,36,21,36,42,
    28,51,36,32,23,28,
    // sum = 500
  };
  const vector<unsigned> expected_lengths {
     8, 6, 9,10,10,16, 7,12, 7, 6,
     9, 5, 7, 8,11, 9,
  };

  nmtkit::WordVocabulary src_vocab, trg_vocab;
  globals::loadArchive(globals::src_vocab_filename, &src_vocab);
  globals::loadArchive(globals::trg_vocab_filename, &trg_vocab);

  nmtkit::SortedRandomSampler sampler(
      globals::src_tok_filename, globals::trg_tok_filename,
      src_vocab, trg_vocab,
      "source_word", "source_target", 256,
      globals::max_length, globals::max_length_ratio, globals::random_seed);

  vector<unsigned> batch_sizes;
  vector<unsigned> lengths;

  while (sampler.hasSamples()) {
    vector<nmtkit::Sample> samples = sampler.getSamples();
    batch_sizes.emplace_back(samples.size());
    unsigned max_length = 0;
    for (auto sample : samples) {
      const unsigned cur_length = sample.source.size();
      max_length = max(max_length, cur_length);
    }
    lengths.emplace_back(max_length);
  }
  BOOST_CHECK_EQUAL_COLLECTIONS(
      expected_batch_sizes.begin(), expected_batch_sizes.end(),
      batch_sizes.begin(), batch_sizes.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(
      expected_lengths.begin(), expected_lengths.end(),
      lengths.begin(), lengths.end());
}

BOOST_AUTO_TEST_CASE(CheckBatch_TargetWord) {
  const vector<unsigned> expected_batch_sizes {
    21,28,21,18,18,16,32,17,19,19,
    36,23,15,25,25,17,23,18,16,21,
    28,19,25,
    // sum = 500
  };
  const vector<unsigned> expected_lengths {
    12, 9,12,14,14,16, 8,15,13,13,
     7,11,16,10,10,15,11,14,16,12,
     9,13,10,
  };

  nmtkit::WordVocabulary src_vocab, trg_vocab;
  globals::loadArchive(globals::src_vocab_filename, &src_vocab);
  globals::loadArchive(globals::trg_vocab_filename, &trg_vocab);

  nmtkit::SortedRandomSampler sampler(
      globals::src_tok_filename, globals::trg_tok_filename,
      src_vocab, trg_vocab,
      "target_word", "target_source", 256,
      globals::max_length, globals::max_length_ratio, globals::random_seed);

  vector<unsigned> batch_sizes;
  vector<unsigned> lengths;

  while (sampler.hasSamples()) {
    vector<nmtkit::Sample> samples = sampler.getSamples();
    batch_sizes.emplace_back(samples.size());
    unsigned max_length = 0;
    for (auto sample : samples) {
      const unsigned cur_length = sample.target.size();
      max_length = max(max_length, cur_length);
    }
    lengths.emplace_back(max_length);
  }
  BOOST_CHECK_EQUAL_COLLECTIONS(
      expected_batch_sizes.begin(), expected_batch_sizes.end(),
      batch_sizes.begin(), batch_sizes.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(
      expected_lengths.begin(), expected_lengths.end(),
      lengths.begin(), lengths.end());
}

BOOST_AUTO_TEST_CASE(CheckSmallBatch_Both) {
  const vector<unsigned> expected_num_batches {
      0, // dummy
    358, // * 1 = 358 sentences
     68, // * 2 = 136 sentences
      2, // * 3 =   6 sentences
      0, // * 4 =   0 sentences
  };     //       500 sentences in total.

  nmtkit::WordVocabulary src_vocab, trg_vocab;
  globals::loadArchive(globals::src_vocab_filename, &src_vocab);
  globals::loadArchive(globals::trg_vocab_filename, &trg_vocab);

  nmtkit::SortedRandomSampler sampler(
      globals::src_tok_filename, globals::trg_tok_filename,
      src_vocab, trg_vocab,
      "both_word", "source_target", 1 + 2 * globals::max_length,
      globals::max_length, globals::max_length_ratio, globals::random_seed);

  vector<unsigned> obtained_num_batches {0, 0, 0, 0, 0};

  while (sampler.hasSamples()) {
    vector<nmtkit::Sample> samples = sampler.getSamples();
    BOOST_CHECK(samples.size() <= 3 /* 4 */);
    ++obtained_num_batches[samples.size()];
  }
  BOOST_CHECK_EQUAL_COLLECTIONS(
      expected_num_batches.begin(), expected_num_batches.end(),
      obtained_num_batches.begin(), obtained_num_batches.end());
}

BOOST_AUTO_TEST_CASE(CheckSmallBatch_Source) {
  const vector<unsigned> expected_num_batches {
      0, // dummy
    167, // * 1 = 167 sentences
    139, // * 2 = 278 sentences
     13, // * 3 =  39 sentences
      4, // * 4 =  16 sentences
  };     //       500 sentences in total.

  nmtkit::WordVocabulary src_vocab, trg_vocab;
  globals::loadArchive(globals::src_vocab_filename, &src_vocab);
  globals::loadArchive(globals::trg_vocab_filename, &trg_vocab);

  nmtkit::SortedRandomSampler sampler(
      globals::src_tok_filename, globals::trg_tok_filename,
      src_vocab, trg_vocab,
      "source_word", "source_target", globals::max_length,
      globals::max_length, globals::max_length_ratio, globals::random_seed);

  vector<unsigned> obtained_num_batches {0, 0, 0, 0, 0};

  while (sampler.hasSamples()) {
    vector<nmtkit::Sample> samples = sampler.getSamples();
    BOOST_CHECK(samples.size() <= 4);
    ++obtained_num_batches[samples.size()];
  }
  BOOST_CHECK_EQUAL_COLLECTIONS(
      expected_num_batches.begin(), expected_num_batches.end(),
      obtained_num_batches.begin(), obtained_num_batches.end());
}

BOOST_AUTO_TEST_CASE(CheckSmallBatch_Target) {
  const vector<unsigned> expected_num_batches {
      0, // dummy
    416, // * 1 = 416 sentences
     39, // * 2 =  78 sentences
      2, // * 3 =   6 sentences
      0, // * 4 =   0 sentences
  };     //       500 sentences in total.

  nmtkit::WordVocabulary src_vocab, trg_vocab;
  globals::loadArchive(globals::src_vocab_filename, &src_vocab);
  globals::loadArchive(globals::trg_vocab_filename, &trg_vocab);

  nmtkit::SortedRandomSampler sampler(
      globals::src_tok_filename, globals::trg_tok_filename,
      src_vocab, trg_vocab,
      "target_word", "target_source", globals::max_length,
      globals::max_length, globals::max_length_ratio, globals::random_seed);

  vector<nmtkit::Sample> samples;
  vector<unsigned> obtained_num_batches {0, 0, 0, 0, 0};

  while (sampler.hasSamples()) {
    vector<nmtkit::Sample> samples = sampler.getSamples();
    BOOST_CHECK(samples.size() <= 3 /* 4 */);
    ++obtained_num_batches[samples.size()];
  }
  BOOST_CHECK_EQUAL_COLLECTIONS(
      expected_num_batches.begin(), expected_num_batches.end(),
      obtained_num_batches.begin(), obtained_num_batches.end());
}

BOOST_AUTO_TEST_SUITE_END()

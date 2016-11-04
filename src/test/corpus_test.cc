#include "config.h"

#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>

#include <algorithm>
#include <fstream>
#include <vector>
#include <boost/archive/text_iarchive.hpp>
#include <nmtkit/corpus.h>
#include <nmtkit/word_vocabulary.h>

using namespace std;

namespace {

const string src_tok_filename = "data/small.en.tok";
const string trg_tok_filename = "data/small.ja.tok";
const string src_vocab_filename = "data/small.en.vocab";
const string trg_vocab_filename = "data/small.ja.vocab";

template <class T>
void loadArchive(const string & filepath, T * obj) {
  ifstream ifs(filepath);
  boost::archive::text_iarchive iar(ifs);
  iar >> *obj;
}

}  // namespace

BOOST_AUTO_TEST_SUITE(CorpusTest)

BOOST_AUTO_TEST_CASE(CheckLoadingSingle) {
  const unsigned expected_num_sents = 500;
  const unsigned expected_num_words = 3871;
  const vector<vector<unsigned>> expected_words {
    {6, 41, 17, 90, 106, 37, 0, 364, 3},
    {159, 0, 13, 130, 0, 101, 332, 3},
    {6, 75, 12, 4, 145, 0, 3},
    {0, 219, 228, 3},
  };

  nmtkit::WordVocabulary vocab;
  ::loadArchive(::src_vocab_filename, &vocab);
  vector<vector<unsigned>> result;
  nmtkit::Corpus::loadSingleSentences(::src_tok_filename, vocab, &result);
  
  BOOST_CHECK_EQUAL(expected_num_sents, result.size());
  
  unsigned num_words = 0;
  for (const auto & sent : result) {
    num_words += sent.size();
  }
  BOOST_CHECK_EQUAL(expected_num_words, num_words);
  
  for (unsigned i = 0; i < expected_words.size(); ++i) {
    BOOST_CHECK_EQUAL_COLLECTIONS(
        expected_words[i].begin(), expected_words[i].end(),
        result[i].begin(), result[i].end());
  }
}

BOOST_AUTO_TEST_CASE(CheckLoadingParallel) {
  const vector<unsigned> max_lengths {2, 4, 8, 16};
  const vector<float> max_length_ratios {1.1, 1.3, 3.0};
  const vector<vector<unsigned>> expected_num_sents {
    {0, 0, 0},
    {1, 1, 1},
    {20, 44, 78},
    {50, 161, 500},
  };
  const vector<vector<unsigned>> expected_num_src_words {
    {0, 0, 0},
    {4, 4, 4},
    {133, 289, 465},
    {447, 1396, 3871},
  };
  const vector<vector<unsigned>> expected_num_trg_words {
    {0, 0, 0},
    {4, 4, 4},
    {133, 297, 552},
    {450, 1552, 5626}
  };

  nmtkit::WordVocabulary src_vocab, trg_vocab;
  ::loadArchive(::src_vocab_filename, &src_vocab);
  ::loadArchive(::trg_vocab_filename, &trg_vocab);
  vector<vector<unsigned>> src_result, trg_result;

  for (unsigned i = 0; i < max_lengths.size(); ++i) {
    for (unsigned j = 0; j < max_length_ratios.size(); ++j) {
      nmtkit::Corpus::loadParallelSentences(
          ::src_tok_filename, ::trg_tok_filename,
          src_vocab, trg_vocab, max_lengths[i], max_length_ratios[j],
          &src_result, &trg_result);

      BOOST_CHECK_EQUAL(src_result.size(), trg_result.size());
      BOOST_CHECK_EQUAL(expected_num_sents[i][j], src_result.size());

      unsigned num_src_words = 0, num_trg_words = 0;
      for (unsigned n = 0; n < src_result.size(); ++n) {
        const unsigned sl = src_result[n].size();
        const unsigned tl = trg_result[n].size();
        BOOST_CHECK(sl <= max_lengths[i]);
        BOOST_CHECK(tl <= max_lengths[i]);
        BOOST_CHECK(max(sl, tl) <= min(sl, tl) * max_length_ratios[j]);
        num_src_words += src_result[n].size();
        num_trg_words += trg_result[n].size();
      }
      BOOST_CHECK_EQUAL(expected_num_src_words[i][j], num_src_words);
      BOOST_CHECK_EQUAL(expected_num_trg_words[i][j], num_trg_words);
    }
  }
}

BOOST_AUTO_TEST_CASE(CheckLoadingParallel2) {
  const vector<unsigned> max_lengths {2, 4, 8, 16};
  const vector<float> max_length_ratios {1.1, 1.3, 3.0};
  const vector<vector<unsigned>> expected_num_sents {
    {0, 0, 0},
    {1, 1, 1},
    {20, 44, 78},
    {50, 161, 500},
  };
  const vector<vector<unsigned>> expected_num_src_words {
    {0, 0, 0},
    {4, 4, 4},
    {133, 289, 465},
    {447, 1396, 3871},
  };
  const vector<vector<unsigned>> expected_num_trg_words {
    {0, 0, 0},
    {4, 4, 4},
    {133, 297, 552},
    {450, 1552, 5626}
  };

  nmtkit::WordVocabulary src_vocab, trg_vocab;
  ::loadArchive(::src_vocab_filename, &src_vocab);
  ::loadArchive(::trg_vocab_filename, &trg_vocab);
  vector<nmtkit::Sample> result;

  for (unsigned i = 0; i < max_lengths.size(); ++i) {
    for (unsigned j = 0; j < max_length_ratios.size(); ++j) {
      nmtkit::Corpus::loadParallelSentences(
          ::src_tok_filename, ::trg_tok_filename,
          src_vocab, trg_vocab, max_lengths[i], max_length_ratios[j],
          &result);

      BOOST_CHECK_EQUAL(expected_num_sents[i][j], result.size());

      unsigned num_src_words = 0, num_trg_words = 0;
      for (unsigned n = 0; n < result.size(); ++n) {
        const unsigned sl = result[n].source.size();
        const unsigned tl = result[n].target.size();
        BOOST_CHECK(sl <= max_lengths[i]);
        BOOST_CHECK(tl <= max_lengths[i]);
        BOOST_CHECK(max(sl, tl) <= min(sl, tl) * max_length_ratios[j]);
        num_src_words += result[n].source.size();
        num_trg_words += result[n].target.size();
      }
      BOOST_CHECK_EQUAL(expected_num_src_words[i][j], num_src_words);
      BOOST_CHECK_EQUAL(expected_num_trg_words[i][j], num_trg_words);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()

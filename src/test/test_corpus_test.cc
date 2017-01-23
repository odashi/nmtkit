#include "config.h"

#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>

#include <algorithm>
#include <fstream>
#include <vector>
#include <boost/archive/text_iarchive.hpp>
#include <nmtkit/test_corpus.h>
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
  boost::archive::binary_iarchive iar(ifs);
  iar >> *obj;
}

}  // namespace

BOOST_AUTO_TEST_SUITE(TestCorpusTest)

BOOST_AUTO_TEST_CASE(CheckLoadingSingle) {
  const unsigned expected_num_sents = 500;
  const unsigned expected_num_words = 3871;
  const vector<vector<unsigned>> expected_words {
    {6, 41, 17, 90, 106, 37, 0, 364, 3},
    {159, 0, 13, 130, 0, 101, 332, 3},
    {6, 75, 12, 4, 145, 0, 3},
    {0, 219, 228, 3},
  };
  const vector<string> expected_strings {
    "i can 't tell who will arrive first .",
    "many animals have been destroyed by men .",
    "i 'm in the tennis club .",
    "emi looks happy ."
  };

  nmtkit::WordVocabulary vocab;
  ::loadArchive(::src_vocab_filename, &vocab);
  vector<vector<unsigned>> result;
  vector<string> result_string;
  nmtkit::TestCorpus::loadSingleSentences(::src_tok_filename, vocab, &result, &result_string);

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
    BOOST_CHECK_EQUAL(expected_strings[i], result_string[i]);
  }
}

BOOST_AUTO_TEST_CASE(CheckLoadingParallel) {
  const vector<vector<unsigned>> expected_src_words {
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

  const vector<vector<unsigned>> expected_trg_words {
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

  nmtkit::WordVocabulary src_vocab, trg_vocab;
  ::loadArchive(::src_vocab_filename, &src_vocab);
  ::loadArchive(::trg_vocab_filename, &trg_vocab);
  vector<vector<unsigned>> src_result, trg_result;
  vector<string> src_string_result;
  vector<string> trg_string_result;

  nmtkit::TestCorpus::loadParallelSentences(
      ::src_tok_filename, ::trg_tok_filename,
      src_vocab, trg_vocab,
      &src_result, &trg_result,
      &src_string_result, &trg_string_result);

  BOOST_CHECK_EQUAL(src_result.size(), trg_result.size());
  BOOST_CHECK_EQUAL(src_string_result.size(), trg_string_result.size());

  for (unsigned i = 0; i < expected_src_words.size(); ++i) {
    BOOST_CHECK_EQUAL_COLLECTIONS(
        expected_src_words[i].begin(), expected_src_words[i].end(),
        src_result[i].begin(), src_result[i].end());
    BOOST_CHECK_EQUAL(expected_src_strings[i], src_string_result[i]);
    BOOST_CHECK_EQUAL_COLLECTIONS(
        expected_trg_words[i].begin(), expected_trg_words[i].end(),
        trg_result[i].begin(), trg_result[i].end());
    BOOST_CHECK_EQUAL(expected_trg_strings[i], trg_string_result[i]);
  }
}

BOOST_AUTO_TEST_CASE(CheckLoadingParallel2) {
  const vector<vector<unsigned>> expected_src_words {
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

  const vector<vector<unsigned>> expected_trg_words {
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

  nmtkit::WordVocabulary src_vocab, trg_vocab;
  ::loadArchive(::src_vocab_filename, &src_vocab);
  ::loadArchive(::trg_vocab_filename, &trg_vocab);
  vector<nmtkit::TestSample> result;

  nmtkit::TestCorpus::loadParallelSentences(
      ::src_tok_filename, ::trg_tok_filename,
      src_vocab, trg_vocab,
      &result);

  for (unsigned i = 0; i < expected_src_words.size(); ++i) {
    BOOST_CHECK_EQUAL_COLLECTIONS(
        expected_src_words[i].begin(), expected_src_words[i].end(),
        result[i].source.begin(), result[i].source.end());
    BOOST_CHECK_EQUAL(expected_src_strings[i], result[i].source_string);
    BOOST_CHECK_EQUAL_COLLECTIONS(
        expected_trg_words[i].begin(), expected_trg_words[i].end(),
        result[i].target.begin(), result[i].target.end());
    BOOST_CHECK_EQUAL(expected_trg_strings[i], result[i].target_string);
  }
}

BOOST_AUTO_TEST_SUITE_END()

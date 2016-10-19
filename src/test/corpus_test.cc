#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>

#include <vector>
#include <nmtkit/corpus.h>
#include <nmtkit/vocabulary.h>

using namespace std;

namespace {

const string src_tok_filename = "data/small.en.tok";
const string trg_tok_filename = "data/small.ja.tok";
const string src_vocab_filename = "data/small.en.vocab";
const string trg_vocab_filename = "data/small.ja.vocab";

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

  nmtkit::Vocabulary vocab(::src_vocab_filename);
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
  const vector<unsigned> expected_num_sents {0, 1, 78, 500};
  const vector<unsigned> expected_num_src_words {0, 4, 465, 3871};
  const vector<unsigned> expected_num_trg_words {0, 4, 552, 5626};

  nmtkit::Vocabulary src_vocab(::src_vocab_filename);
  nmtkit::Vocabulary trg_vocab(::trg_vocab_filename);
  vector<vector<unsigned>> src_result, trg_result;

  for (unsigned i = 0; i < max_lengths.size(); ++i) {
    nmtkit::Corpus::loadParallelSentences(
        ::src_tok_filename, ::trg_tok_filename,
        src_vocab, trg_vocab, max_lengths[i],
        &src_result, &trg_result);

    BOOST_CHECK_EQUAL(src_result.size(), trg_result.size());
    BOOST_CHECK_EQUAL(expected_num_sents[i], src_result.size());

    unsigned num_src_words = 0, num_trg_words = 0;
    for (unsigned j = 0; j < src_result.size(); ++j) {
      BOOST_CHECK(src_result[j].size() <= max_lengths[i]);
      BOOST_CHECK(trg_result[j].size() <= max_lengths[i]);
      num_src_words += src_result[j].size();
      num_trg_words += trg_result[j].size();
    }
    BOOST_CHECK_EQUAL(expected_num_src_words[i], num_src_words);
    BOOST_CHECK_EQUAL(expected_num_trg_words[i], num_trg_words);
  }
}

BOOST_AUTO_TEST_CASE(CheckLoadingParallel2) {
  const vector<unsigned> max_lengths {2, 4, 8, 16};
  const vector<unsigned> expected_num_sents {0, 1, 78, 500};
  const vector<unsigned> expected_num_src_words {0, 4, 465, 3871};
  const vector<unsigned> expected_num_trg_words {0, 4, 552, 5626};

  nmtkit::Vocabulary src_vocab(::src_vocab_filename);
  nmtkit::Vocabulary trg_vocab(::trg_vocab_filename);
  vector<nmtkit::Sample> result;

  for (unsigned i = 0; i < max_lengths.size(); ++i) {
    nmtkit::Corpus::loadParallelSentences(
        ::src_tok_filename, ::trg_tok_filename,
        src_vocab, trg_vocab, max_lengths[i],
        &result);

    BOOST_CHECK_EQUAL(expected_num_sents[i], result.size());

    unsigned num_src_words = 0, num_trg_words = 0;
    for (unsigned j = 0; j < result.size(); ++j) {
      BOOST_CHECK(result[j].source.size() <= max_lengths[i]);
      BOOST_CHECK(result[j].target.size() <= max_lengths[i]);
      num_src_words += result[j].source.size();
      num_trg_words += result[j].target.size();
    }
    BOOST_CHECK_EQUAL(expected_num_src_words[i], num_src_words);
    BOOST_CHECK_EQUAL(expected_num_trg_words[i], num_trg_words);
  }
}

BOOST_AUTO_TEST_SUITE_END()


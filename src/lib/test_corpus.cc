#include "config.h"

#include <nmtkit/test_corpus.h>

#include <algorithm>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <nmtkit/exception.h>

using namespace std;

namespace nmtkit {

bool TestCorpus::readLine(istream * is, string * line) {
  if (!getline(*is, *line)) return false;
  boost::trim(*line);
  return true;
}

bool TestCorpus::readTokens(
    const Vocabulary & vocab,
    istream * is,
    vector<unsigned> * word_ids,
    string * sent_string) {
  string line;
  if (!readLine(is, &line)) return false;
  *word_ids = vocab.convertToIDs(line);
  *sent_string = line;
  return true;
}

void TestCorpus::loadSingleSentences(
    const string & filepath,
    const Vocabulary & vocab,
    vector<vector<unsigned>> * result, 
    vector<string> * string_result) {
  ifstream ifs(filepath);
  NMTKIT_CHECK(
      ifs.is_open(), "Could not open the corpus file to load: " + filepath);

  result->clear();
  vector<unsigned> word_ids;
  string sent_string;
  while (readTokens(vocab, &ifs, &word_ids, &sent_string)) {
    result->emplace_back(std::move(word_ids));
    string_result->emplace_back(std::move(sent_string));
  }
}

void TestCorpus::loadParallelSentences(
    const string & src_filepath,
    const string & trg_filepath,
    const Vocabulary & src_vocab,
    const Vocabulary & trg_vocab,
    vector<vector<unsigned>> * src_result,
    vector<vector<unsigned>> * trg_result,
    vector<string> * src_string_result,
    vector<string> * trg_string_result) {
  ifstream src_ifs(src_filepath), trg_ifs(trg_filepath);
  NMTKIT_CHECK(
      src_ifs.is_open(),
      "Could not open the source corpus file to load: " + src_filepath);
  NMTKIT_CHECK(
      trg_ifs.is_open(),
      "Could not open the target corpus file to load: " + trg_filepath);

  src_result->clear();
  trg_result->clear();
  src_string_result->clear();
  trg_string_result->clear();
  vector<unsigned> src_ids, trg_ids;
  string src_string, trg_string;
  while (
      readTokens(src_vocab, &src_ifs, &src_ids, &src_string) and
      readTokens(trg_vocab, &trg_ifs, &trg_ids, &trg_string)) {
    src_result->emplace_back(std::move(src_ids));
    trg_result->emplace_back(std::move(trg_ids));
    src_string_result->emplace_back(std::move(src_string));
    trg_string_result->emplace_back(std::move(trg_string));
  }
}

void TestCorpus::loadParallelSentences(
    const string & src_filepath,
    const string & trg_filepath,
    const Vocabulary & src_vocab,
    const Vocabulary & trg_vocab,
    vector<TestSample> * result) {
  ifstream src_ifs(src_filepath), trg_ifs(trg_filepath);
  NMTKIT_CHECK(
      src_ifs.is_open(),
      "Could not open the source corpus file to load: " + src_filepath);
  NMTKIT_CHECK(
      trg_ifs.is_open(),
      "Could not open the target corpus file to load: " + trg_filepath);

  result->clear();
  vector<unsigned> src_ids, trg_ids;
  string src_string, trg_string;
  while (
      readTokens(src_vocab, &src_ifs, &src_ids, &src_string) and
      readTokens(trg_vocab, &trg_ifs, &trg_ids, &trg_string)) {
    result->emplace_back(TestSample {std::move(src_ids), std::move(trg_ids),
            std::move(src_string), std::move(trg_string)});
  }
}

}  // namespace nmtkit

#include <nmtkit/corpus.h>

#include <algorithm>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <nmtkit/exception.h>

using std::ifstream;
using std::istream;
using std::string;
using std::swap;
using std::vector;

namespace {

// Checks whether given sample is acceptable or not.
//
// Arguments:
//   src_ids: List of source word IDs.
//   trg_ids: List of target word IDs.
//   max_length: Acceptable maximum of the sequence length.
//   max_length_ratio: Acceptable maximum of the ratio of both sequences
//                     lengths.
//
// Returns:
//   true if the sample is acceptable, false otherwise.
bool checkSample(
    const vector<unsigned> & src_ids,
    const vector<unsigned> & trg_ids,
    unsigned max_length,
    float max_length_ratio) {
  unsigned len1 = src_ids.size();
  unsigned len2 = trg_ids.size();
  if (len1 > max_length || len2 > max_length) return false;
  if (len1 > len2) swap(len1, len2);
  if (len2 > len1 * max_length_ratio) return false;
  return true;
}

}  // namespae

namespace nmtkit {

bool Corpus::readLine(istream * is, string * line) {
  if (!getline(*is, *line)) return false;
  boost::trim(*line);
  return true;
}

bool Corpus::readTokens(
    const Vocabulary & vocab,
    istream * is,
    vector<unsigned> * word_ids) {
  string line;
  if (!readLine(is, &line)) return false;
  *word_ids = vocab.convertToIDs(line);
  return true;
}

void Corpus::loadSingleSentences(
    const string & filepath,
    const Vocabulary & vocab,
    vector<vector<unsigned>> * result) {
  ifstream ifs(filepath);
  NMTKIT_CHECK(
      ifs.is_open(), "Could not open the corpus file to load: " + filepath);

  result->clear();
  vector<unsigned> word_ids;
  while (readTokens(vocab, &ifs, &word_ids)) {
    result->emplace_back(std::move(word_ids));
  }
}

void Corpus::loadParallelSentences(
    const string & src_filepath,
    const string & trg_filepath,
    const Vocabulary & src_vocab,
    const Vocabulary & trg_vocab,
    unsigned max_length,
    float max_length_ratio,
    vector<vector<unsigned>> * src_result,
    vector<vector<unsigned>> * trg_result) {
  ifstream src_ifs(src_filepath), trg_ifs(trg_filepath);
  NMTKIT_CHECK(
      src_ifs.is_open(),
      "Could not open the source corpus file to load: " + src_filepath);
  NMTKIT_CHECK(
      trg_ifs.is_open(),
      "Could not open the target corpus file to load: " + trg_filepath);
  NMTKIT_CHECK(max_length > 0, "max_length should be greater than 0.");

  src_result->clear();
  trg_result->clear();
  vector<unsigned> src_ids, trg_ids;
  while (
      readTokens(src_vocab, &src_ifs, &src_ids) and
      readTokens(trg_vocab, &trg_ifs, &trg_ids)) {
    if (::checkSample(src_ids, trg_ids, max_length, max_length_ratio)) {
      src_result->emplace_back(std::move(src_ids));
      trg_result->emplace_back(std::move(trg_ids));
    }
  }
}

void Corpus::loadParallelSentences(
    const string & src_filepath,
    const string & trg_filepath,
    const Vocabulary & src_vocab,
    const Vocabulary & trg_vocab,
    unsigned max_length,
    float max_length_ratio,
    vector<Sample> * result) {
  ifstream src_ifs(src_filepath), trg_ifs(trg_filepath);
  NMTKIT_CHECK(
      src_ifs.is_open(),
      "Could not open the source corpus file to load: " + src_filepath);
  NMTKIT_CHECK(
      trg_ifs.is_open(),
      "Could not open the target corpus file to load: " + trg_filepath);
  NMTKIT_CHECK(max_length > 0, "max_length should be greater than 0.");

  result->clear();
  vector<unsigned> src_ids, trg_ids;
  while (
      readTokens(src_vocab, &src_ifs, &src_ids) and
      readTokens(trg_vocab, &trg_ifs, &trg_ids)) {
    if (::checkSample(src_ids, trg_ids, max_length, max_length_ratio)) {
      result->emplace_back(Sample {std::move(src_ids), std::move(trg_ids)});
    }
  }
}

}  // namespace nmtkit

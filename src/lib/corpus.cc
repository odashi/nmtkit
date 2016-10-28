#include <nmtkit/corpus.h>

#include <algorithm>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <nmtkit/exception.h>

using namespace std;

namespace nmtkit {

bool Corpus::readTokens(istream * is, vector<string> * words) {
  string line;
  if (!getline(*is, line)) {
    return false;
  }
  boost::trim(line);
  boost::split(
      *words, line, boost::is_space(), boost::algorithm::token_compress_on);
  return true;
}

void Corpus::wordsToWordIDs(
    const vector<string> & words,
    const nmtkit::Vocabulary & vocab,
    vector<unsigned> * ids) {
  ids->resize(words.size());
  for (unsigned i = 0; i < words.size(); ++i) {
    (*ids)[i] = vocab.getID(words[i]);
  }
}

void Corpus::loadSingleSentences(
    const string & filepath,
    const Vocabulary & vocab,
    vector<vector<unsigned>> * result) {
  ifstream ifs(filepath);
  NMTKIT_CHECK(
      ifs.is_open(), "Could not open the corpus file to load: " + filepath);

  result->clear();
  vector<string> words;
  while (readTokens(&ifs, &words)) {
    result->emplace_back(vector<unsigned>());
    wordsToWordIDs(words, vocab, &result->back());
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
  vector<string> src_words, trg_words;
  while (
      readTokens(&src_ifs, &src_words) &&
      readTokens(&trg_ifs, &trg_words)) {

    // Filters sentences.
    unsigned len1 = src_words.size();
    unsigned len2 = trg_words.size();
    if (len1 > max_length || len2 > max_length) {
      continue;
    }
    if (len1 > len2) {
      swap(len1, len2);
    }
    if (len2 > len1 * max_length_ratio) {
      continue;
    }

    src_result->emplace_back(vector<unsigned>());
    trg_result->emplace_back(vector<unsigned>());
    wordsToWordIDs(src_words, src_vocab, &src_result->back());
    wordsToWordIDs(trg_words, trg_vocab, &trg_result->back());
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
  vector<string> src_words, trg_words;
  while (
      readTokens(&src_ifs, &src_words) &&
      readTokens(&trg_ifs, &trg_words)) {

    // Filters sentences.
    unsigned len1 = src_words.size();
    unsigned len2 = trg_words.size();
    if (len1 > max_length || len2 > max_length) {
      continue;
    }
    if (len1 > len2) {
      swap(len1, len2);
    }
    if (len2 > len1 * max_length_ratio) {
      continue;
    }


    result->emplace_back(Sample());
    wordsToWordIDs(src_words, src_vocab, &result->back().source);
    wordsToWordIDs(trg_words, trg_vocab, &result->back().target);
  }
}

}  // namespace nmtkit

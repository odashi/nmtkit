#include <nmtkit/corpus.h>

#include <fstream>
#include <boost/algorithm/string.hpp>
#include <nmtkit/exception.h>

using namespace std;

namespace {

// Reads one line from the input stream and split into words.
bool readTokens(ifstream * ifs, vector<string> * words) {
  string line;
  if (!getline(*ifs, line)) {
    return false;
  }
  boost::trim(line);
  boost::split(
      *words, line, boost::is_space(), boost::algorithm::token_compress_on);
  return true;
}

// Converts words into word-IDs.
void convertToIDs(
    const vector<string> & words,
    const nmtkit::Vocabulary & vocab,
    vector<unsigned> * ids) {
  ids->resize(words.size());
  for (unsigned i = 0; i < words.size(); ++i) {
    (*ids)[i] = vocab.getID(words[i]);
  }
}

}  // namespace

namespace nmtkit {

void Corpus::loadSingleSentences(
    const string & filepath,
    const Vocabulary & vocab,
    vector<vector<unsigned>> * result) {
  ifstream ifs(filepath);
  NMTKIT_CHECK(
      ifs.is_open(), "Could not open the corpus file to load: " + filepath);

  result->clear();
  vector<string> words;
  while (::readTokens(&ifs, &words)) {
    result->emplace_back(vector<unsigned>());
    ::convertToIDs(words, vocab, &result->back());
  }
}

void Corpus::loadParallelSentences(
    const string & src_filepath,
    const string & trg_filepath,
    const Vocabulary & src_vocab,
    const Vocabulary & trg_vocab,
    unsigned max_length,
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
      ::readTokens(&src_ifs, &src_words) &&
      ::readTokens(&trg_ifs, &trg_words)) {

    // Filters sentences.
    if (src_words.size() > max_length || trg_words.size() > max_length) {
      continue;
    }

    src_result->emplace_back(vector<unsigned>());
    trg_result->emplace_back(vector<unsigned>());
    ::convertToIDs(src_words, src_vocab, &src_result->back());
    ::convertToIDs(trg_words, trg_vocab, &trg_result->back());
  }
}

}  // namespace nmtkit


#include "config.h"

#include <nmtkit/word_vocabulary.h>

#include <fstream>
#include <functional>
#include <boost/algorithm/string.hpp>
#include <nmtkit/array.h>
#include <nmtkit/corpus.h>
#include <nmtkit/exception.h>

using namespace std;

namespace nmtkit {

WordVocabulary::WordVocabulary(const string & corpus_filename, unsigned size) {
  NMTKIT_CHECK(size >= 3, "Size should be equal or greater than 3.");
  ifstream ifs(corpus_filename);
  NMTKIT_CHECK(
      ifs.is_open(),
      "Could not open corpus file to load: " + corpus_filename);

  // Counts word frequencies.
  map<string, unsigned> freq;
  string line;
  unsigned num_lines = 0;
  unsigned num_words = 0;
  while (Corpus::readLine(&ifs, &line)) {
    ++num_lines;
    vector<string> words;
    boost::split(
        words, line, boost::is_space(), boost::algorithm::token_compress_on);
    num_words += words.size();
    for (const string & word : words) {
      ++freq[word];
    }
  }

  // Selects most frequent words.
  vector<pair<unsigned, string>> entries;
  for (const auto & entry : freq) {
    entries.emplace_back(make_pair(entry.second, entry.first));
  }
  Array::sort(&entries, greater<pair<unsigned, string>>());

  // Store entries.
  stoi_["<unk>"] = 0;
  stoi_["<s>"] = 1;
  stoi_["</s>"] = 2;
  itos_.emplace_back("<unk>");
  itos_.emplace_back("<s>");
  itos_.emplace_back("</s>");
  freq_.emplace_back(num_words);
  freq_.emplace_back(num_lines);
  freq_.emplace_back(num_lines);
  for (unsigned i = 3; i < size && i - 3 < entries.size(); ++i) {
    const auto & entry = entries[i - 3];
    stoi_[entry.second] = i;
    itos_.emplace_back(entry.second);
    freq_.emplace_back(entry.first);
    freq_[0] -= entry.first;
  }
}

unsigned WordVocabulary::getID(const string & word) const {
  const auto &entry = stoi_.find(word);
  if (entry == stoi_.end()) return 0;  // ID of <unk>
  return entry->second;
}

string WordVocabulary::getWord(const unsigned id) const {
  NMTKIT_CHECK(id < itos_.size(), "Index out of range.");
  return itos_[id];
}

unsigned WordVocabulary::getFrequency(const unsigned id) const {
  NMTKIT_CHECK(id < itos_.size(), "Index out of range.");
  return freq_[id];
}

vector<unsigned> WordVocabulary::convertToIDs(const string & sentence) const {
  vector<string> words;
  boost::split(
      words, sentence, boost::is_space(), boost::algorithm::token_compress_on);
  vector<unsigned> ids;
  for (const string & word : words) {
    ids.emplace_back(getID(word));
  }
  return ids;
}

string WordVocabulary::convertToSentence(
    const vector<unsigned> & word_ids) const {
  vector<string> words;
  for (const unsigned word_id : word_ids) {
    words.emplace_back(getWord(word_id));
  }
  return boost::join(words, " ");
}

unsigned WordVocabulary::size() const {
  return itos_.size();
}

}  // namespace nmtkit

NMTKIT_SERIALIZATION_IMPL(nmtkit::WordVocabulary);

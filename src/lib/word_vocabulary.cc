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
  while (Corpus::readLine(&ifs, &line)) {
    vector<string> words;
    boost::split(
        words, line, boost::is_space(), boost::algorithm::token_compress_on);
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
  for (unsigned i = 3; i < size && i - 3 < entries.size(); ++i) {
    const string & word = entries[i - 3].second;
    stoi_[word] = i;
    itos_.emplace_back(word);
  }
}

unsigned WordVocabulary::getID(const string & word) const {
  const auto &entry = stoi_.find(word);
  if (entry == stoi_.end()) return 0;  // ID of <unk>
  return entry->second;
}

string WordVocabulary::getWord(unsigned id) const {
  if (id >= itos_.size()) return "<unk>";  // out of range
  return itos_[id];
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

#include "config.h"

#include <nmtkit/character_vocabulary.h>

#include <fstream>
#include <functional>
#include <boost/algorithm/string.hpp>
#include <nmtkit/array.h>
#include <nmtkit/corpus.h>
#include <nmtkit/exception.h>
#include <nmtkit/unicode.h>

using nmtkit::UTF8;
using std::greater;
using std::ifstream;
using std::map;
using std::pair;
using std::string;
using std::vector;

namespace nmtkit {

CharacterVocabulary::CharacterVocabulary(
    const string & corpus_filename,
    unsigned size) {
  NMTKIT_CHECK(size >= 3, "Size should be equal or greater than 3.");
  ifstream ifs(corpus_filename);
  NMTKIT_CHECK(
      ifs.is_open(),
      "Could not open corpus file to load: " + corpus_filename);

  // Counts letter frequencies.
  map<string, unsigned> freq;
  string line;
  unsigned num_lines = 0;
  unsigned num_letters = 0;
  while (Corpus::readLine(&ifs, &line)) {
    ++num_lines;
    const vector<string> letters = UTF8::getLetters(line);
    num_letters += letters.size();
    for (const string & letter : letters) {
      ++freq[letter];
    }
  }

  // Selects most frequent letters.
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
  freq_.emplace_back(num_letters);
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

unsigned CharacterVocabulary::getID(const string & word) const {
  const auto &entry = stoi_.find(word);
  if (entry == stoi_.end()) return 0;  // ID of <unk>
  return entry->second;
}

string CharacterVocabulary::getWord(const unsigned id) const {
  NMTKIT_CHECK(id < itos_.size(), "Index out of range.");
  return itos_[id];
}

unsigned CharacterVocabulary::getFrequency(const unsigned id) const {
  NMTKIT_CHECK(id < itos_.size(), "Index out of range.");
  return freq_[id];
}

vector<unsigned> CharacterVocabulary::convertToIDs(
    const string & sentence) const {
  vector<unsigned> ids;
  for (const string & letter : UTF8::getLetters(sentence)) {
    ids.emplace_back(getID(letter));
  }
  return ids;
}

string CharacterVocabulary::convertToSentence(
    const vector<unsigned> & word_ids) const {
  vector<string> letters;
  for (const unsigned word_id : word_ids) {
    letters.emplace_back(getWord(word_id));
  }
  return boost::join(letters, "");
}

unsigned CharacterVocabulary::size() const {
  return itos_.size();
}

}  // namespace nmtkit

NMTKIT_SERIALIZATION_IMPL(nmtkit::CharacterVocabulary);

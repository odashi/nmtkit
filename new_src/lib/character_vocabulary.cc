#include <nmtkit/character_vocabulary.h>

#include <config.h>
#include <boost/algorithm/string.hpp>
#include <nmtkit/array.h>
#include <nmtkit/exception.h>
#include <nmtkit/unicode.h>
#include <fstream>
#include <functional>

using nmtkit::UTF8;
using std::greater;
using std::ifstream;
using std::map;
using std::pair;
using std::string;
using std::vector;

namespace nmtkit {

CharacterVocabulary::CharacterVocabulary(Reader * reader, const unsigned size) {
  NMTKIT_CHECK(size >= 4);

  // Counts letter frequencies.
  map<string, unsigned> freq;
  Sentence sent;
  unsigned num_lines = 0;
  unsigned num_spaces = 0;
  unsigned num_letters = 0;
  while (reader->read(&sent)) {
    ++num_lines;
    if (!sent.tokens.empty()) {
      num_spaces += sent.tokens.size() - 1;
    }
    for (const Token & tok : sent.tokens) {
      const vector<string> letters = UTF8::getLetters(tok.surface);
      num_letters += letters.size();
      for (const string & letter : letters) {
        ++freq[letter];
      }
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
  stoi_["<sp>"] = 3;
  itos_.emplace_back("<unk>");
  itos_.emplace_back("<s>");
  itos_.emplace_back("</s>");
  itos_.emplace_back("<sp>");
  freq_.emplace_back(num_letters);
  freq_.emplace_back(num_lines);
  freq_.emplace_back(num_lines);
  freq_.emplace_back(num_spaces);
  for (unsigned i = 4; i < size && i - 4 < entries.size(); ++i) {
    const auto & entry = entries[i - 4];
    stoi_[entry.second] = i;
    itos_.emplace_back(entry.second);
    freq_.emplace_back(entry.first);
    freq_[0] -= entry.first;
  }
}

vector<unsigned> CharacterVocabulary::convertToIDs(const Token & token) const {
  const auto & entry = cache_.find(token.surface);
  if (entry != cache_.end()) {
    return entry->second;
  }

  const vector<string> letters = UTF8::getLetters(token.surface);
  vector<unsigned> ids(letters.size());
  for (unsigned i = 0; i < letters.size(); ++i) {
    const auto & entry = stoi_.find(letters[i]);
    ids[i] = entry != stoi_.end() ? entry->second : 0 /* <unk> */;
  }

  cache_.insert(std::make_pair(token.surface, ids));
  return ids;
}

vector<unsigned> CharacterVocabulary::convertToIDs(
    const Sentence & sentence) const {
  vector<unsigned> ids;
  bool first_time = true;
  for (const Token & tok : sentence.tokens) {
    if (!first_time) {
      ids.emplace_back(3 /* <sp> */);
    }
    first_time = false;

    for (const unsigned id : convertToIDs(tok)) {
      ids.emplace_back(id);
    }
  }
  return ids;
}

Sentence CharacterVocabulary::convertToSentence(
    const vector<unsigned> & ids) const {
  // Empty ID list generates empty sentence.
  if (ids.empty()) {
    return Sentence();
  }

  Sentence sent;
  Token temp_tok;

  for (const unsigned id : ids) {
    if (id == 3 /* <sp> */) {
      // Flush current token at this point.
      // This code may generate empty tokens ("").
      sent.tokens.emplace_back(std::move(temp_tok));
      temp_tok = Token();
    } else {
      temp_tok.surface += getSurface(id);
    }
  }

  sent.tokens.emplace_back(std::move(temp_tok));
  return sent;
}

string CharacterVocabulary::getSurface(const unsigned id) const {
  NMTKIT_CHECK(id < itos_.size());
  return itos_[id];
}

unsigned CharacterVocabulary::getFrequency(const unsigned id) const {
  NMTKIT_CHECK(id < itos_.size());
  return freq_[id];
}

unsigned CharacterVocabulary::size() const {
  return itos_.size();
}

}  // namespace nmtkit

NMTKIT_SERIALIZATION_IMPL(nmtkit::CharacterVocabulary);

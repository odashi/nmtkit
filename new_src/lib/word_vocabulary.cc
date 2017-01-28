#include <nmtkit/word_vocabulary.h>

#include <config.h>
#include <nmtkit/array.h>
#include <nmtkit/exception.h>
#include <utility>

using std::string;
using std::vector;

namespace nmtkit {

WordVocabulary::WordVocabulary(Reader * reader, const unsigned size) {
  NMTKIT_CHECK(size >= 3);

  // Counts word frequencies.
  std::map<string, unsigned> freq;
  Sentence sent;
  unsigned num_lines = 0;
  unsigned num_words = 0;
  while (reader->read(&sent)) {
    ++num_lines;
    num_words += sent.tokens.size();
    for (const Token & tok : sent.tokens) {
      ++freq[tok.surface];
    }
  }

  // Selects most frequent words.
  vector<std::pair<unsigned, string>> entries;
  for (const auto & entry : freq) {
    entries.emplace_back(std::make_pair(entry.second, entry.first));
  }
  Array::sort(&entries, std::greater<std::pair<unsigned, string>>());

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

vector<unsigned> WordVocabulary::convertToIDs(const Token & token) const {
  const auto & entry = stoi_.find(token.surface);
  return { entry != stoi_.end() ? entry->second : 0 /* <unk> */ };
}

vector<unsigned> WordVocabulary::convertToIDs(const Sentence & sentence) const {
  vector<unsigned> ids(sentence.tokens.size());
  for (unsigned i = 0; i < sentence.tokens.size(); ++i) {
    const auto & entry = stoi_.find(sentence.tokens[i].surface);
    ids[i] = entry != stoi_.end() ? entry->second : 0 /* <unk> */;
  }
  return ids;
}

Sentence WordVocabulary::convertToSentence(const vector<unsigned> & ids) const {
  vector<string> surfaces(ids.size());
  for (unsigned i = 0; i < ids.size(); ++i) {
    surfaces[i] = getSurface(ids[i]);
  }
  return Sentence(surfaces);
}

string WordVocabulary::getSurface(const unsigned id) const {
  NMTKIT_CHECK(id < itos_.size());
  return itos_[id];
}

unsigned WordVocabulary::getFrequency(const unsigned id) const {
  NMTKIT_CHECK(id < itos_.size());
  return freq_[id];
}

unsigned WordVocabulary::size() const {
  return itos_.size();
}

}  // namespace nmtkit

NMTKIT_SERIALIZATION_IMPL(nmtkit::WordVocabulary);

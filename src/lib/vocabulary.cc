#include <nmtkit/vocabulary.h>

#include <algorithm>
#include <fstream>
#include <functional>
#include <boost/algorithm/string.hpp>
#include <nmtkit/exception.h>

using namespace std;

namespace NMTKit {

Vocabulary::Vocabulary(const string &vocab_filename) {
  ifstream ifs(vocab_filename);
  NMTKIT_CHECK(
      ifs.is_open(),
      "Could not open vocabulary file to load: " + vocab_filename);

  // Loads vocabulary size.
  int vocab_size;
  ifs >> vocab_size;
  NMTKIT_CHECK(
      vocab_size >= 3, "Vocabulary size should be equal or greater than 3.");

  // Loads each entry.
  for (int i = 0; i < vocab_size; ++i) {
    string word;
    ifs >> word;
    stoi_[word] = i;
    itos_.emplace_back(word);
  }
  NMTKIT_CHECK_EQ(
      "<unk>", itos_[0], "0th entry of the vocabulary should be \"<unk>\".");
  NMTKIT_CHECK_EQ(
      "<s>", itos_[1], "1st entry of the vocabulary should be \"<s>\".");
  NMTKIT_CHECK_EQ(
      "</s>", itos_[2], "2nd entry of the vocabulary should be \"</s>\".");
}

Vocabulary::Vocabulary(const string &corpus_filename, int size) {
  NMTKIT_CHECK(size >= 3, "Size should be equal or greater than 3.");
  ifstream ifs(corpus_filename);
  NMTKIT_CHECK(
      ifs.is_open(),
      "Could not open corpus file to load: " + corpus_filename);

  // Counts word frequencies.
  map<string, int> freq;
  string line;
  while (getline(ifs, line)) {
    boost::trim(line);
    vector<string> words;
    boost::split(
        words, line, boost::is_space(), boost::algorithm::token_compress_on);
    for (const string &word : words) {
      ++freq[word];
    }
  }

  // Selects most frequent words.
  vector<pair<int, string>> entries;
  for (const auto &entry : freq) {
    entries.emplace_back(make_pair(entry.second, entry.first));
  }
  sort(entries.begin(), entries.end(), greater<pair<int, string>>());
  
  // Store entries.
  stoi_["<unk>"] = 0;
  stoi_["<s>"] = 1;
  stoi_["</s>"] = 2;
  itos_.emplace_back("<unk>");
  itos_.emplace_back("<s>");
  itos_.emplace_back("</s>");
  for (int i = 3; i < size && i - 3 < entries.size(); ++i) {
    const string &word = entries[i - 3].second;
    stoi_[word] = i;
    itos_.emplace_back(word);
  }
}

void Vocabulary::save(const string &vocab_filename) {
  ofstream ofs(vocab_filename);
  NMTKIT_CHECK(
      ofs.is_open(),
      "Could not open vocabulary file to save: " + vocab_filename);
  ofs << itos_.size() << endl;
  for (const string &word : itos_) {
    ofs << word << endl;
  }
}

int Vocabulary::getID(const string &word) const {
  const auto &entry = stoi_.find(word);
  if (entry == stoi_.end()) return 0;  // ID of <unk>
  return entry->second;
}

string Vocabulary::getWord(int id) const {
  if (id < 0 or id >= itos_.size()) return "<unk>";  // out of range
  return itos_[id];
}

int Vocabulary::size() const {
  return itos_.size();
}

}  // namespace NMTKit


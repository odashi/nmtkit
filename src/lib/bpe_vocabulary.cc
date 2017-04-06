#include "config.h"

#include <nmtkit/bpe_vocabulary.h>

#include <fstream>
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

namespace {

struct Change {
  unsigned index;
  vector<string> new_word;
  vector<string> old_word;
  int freq;
};

// Calculate character bigram frequency.
//
// Arguments:
//   vocab: vector bigram frequency.
//   stats: sum of bigram frequency.
//   indices: index of stats (key=bigram)
void getPairStatistics(vector<pair<vector<string>, int>> * vocab,
    map<vector<string>, int> * stats,
    map<vector<string>, map<unsigned, int>> * indices) {

  for (unsigned i = 0; i < vocab->size(); i++) {
    const vector<string> word = (*vocab)[i].first;
    const int freq = (*vocab)[i].second;
    string prev_char = word[0];
    for (unsigned j = 1; j < word.size(); j++) {
      const string current_char = word[j];
      const vector<string> key = {prev_char, current_char};
      (*stats)[key] += freq;
      (*indices)[key][i] += 1;
      prev_char = current_char;
    }
  }
}

// Find max frequency in stats.
//
// Arguments:
//   stats: sum of bigram frequency
//
// Returns:
//   most frequent bigram
vector<string> findMax(const map<vector<string>, int> * stats) {
  int current_max = -1e5;
  vector<string> current_argmax;
  for (auto elm : (*stats)) {
    if (elm.second > current_max) {
      current_max = elm.second;
      current_argmax = elm.first;
    }
  }
  return current_argmax;
}

// Find replaceable word pairs.
//
// Arguments:
//   replace_words: words that are going to concatenate
//   vocab: vector vocabulary
//   indices: stats indices of repalce_words
//
// Returns:
//   vector of replaceable pairs
vector<Change> replacePair(const vector<string> * replace_words,
    vector<pair<vector<string>, int>> * vocab,
    const map<unsigned, int> * indices) {
  string first = (*replace_words)[0];
  string second = (*replace_words)[1];
  string pair_str = boost::join((*replace_words), "");
  vector<Change> changes;

  for (const auto index : (*indices)) {
    unsigned j = index.first;
    int freq = index.second;
    if (freq < 1) {
        continue;
    }

    vector<string> word = (*vocab)[j].first;
    freq = (*vocab)[j].second;
    string new_word = boost::join(word, " ");
    boost::replace_all(new_word, first + " " + second, pair_str);
    vector<string> vector_new_word;
    boost::split(
        vector_new_word, new_word, boost::is_space(), boost::algorithm::token_compress_on);
    (*vocab)[j] = pair<vector<string>, int>(vector_new_word, freq);
    changes.emplace_back( Change{j, vector_new_word, word, freq} );
  }

  return changes;
}

// Find index of the specific word from the vector
//
// Arguments:
//   word: vector of words
//   search_word: search query word
//   start_index: start finding from this index
//
// Returns:
//   index of the specific word
int findIndex(vector<string> * word, string * search_word, unsigned start_index) {
  auto iter = find(word->begin() + start_index, word->end(), (*search_word));
  size_t index = distance(word->begin(), iter);
  return index;
}

// Update stats based on changes
//
// Arguments:
//   replace_words: words that are going to concatenate
//   changes: return value of replacePair()
//   stats: sum of bigram frequency.
//   indices: index of stats (key=bigram)
void updatePairStatistics(const vector<string> * replace_words,
    const vector<Change> * changes,
    map<vector<string>, int> * stats,
    map<vector<string>, map<unsigned, int>> * indices) {
  stats->erase((*replace_words));
  indices->erase((*replace_words));
  string first = (*replace_words)[0];
  string second = (*replace_words)[1];
  string new_pair = first + second;

  for (unsigned i = 0; i < changes->size(); i++) {
    unsigned j = (*changes)[i].index;
    vector<string> new_word = (*changes)[i].new_word;
    vector<string> old_word = (*changes)[i].old_word;
    int freq = (*changes)[i].freq;

    unsigned k = 0;
    while(true) {
      k = findIndex(&old_word, &first, k);
      if (k == old_word.size()) {
        break;
      }
      if (k < old_word.size() - 1 and old_word[k+1] == second) {
        if (k != 0) {
          vector<string> prev = {old_word[k-1], old_word[k]};
          (*stats)[prev] -= freq;
          (*indices)[prev][j] -= 1;
        }
        if (k < old_word.size() - 2) {
          if (old_word[k+2] != first or k >= old_word.size() - 3 or old_word[k+3] != second) {
            vector<string> nex = {old_word[k+1], old_word[k+2]};
            (*stats)[nex] -= freq;
            (*indices)[nex][j] -= 1;
          }
        }
        k += 2;
      }
      else {
        k += 1;
      }
    }

    k = 0;
    while(true) {
      k = findIndex(&new_word, &new_pair, k);
      if (k == new_word.size()) {
        break;
      }
      if (k != 0) {
        vector<string> prev = {new_word[k-1], new_word[k]};
        (*stats)[prev] += freq;
        (*indices)[prev][j] += 1;
      }
      if (k < new_word.size() - 1 and new_word[k+1] != new_pair) {
        vector<string> nex = {new_word[k], new_word[k+1]};
        (*stats)[nex] += freq;
        (*indices)[nex][j] += 1;
      }
      k += 1;
    }
  }
}

// Prune low frequency words from stats
//
// Arguments:
//   stats: sum of bigram frequency.
//   big_stats: sum of bigram frequency (not pruned).
//   threshold: words that frequency is less than this threshold will be pruned
void pruneStats(
    map<vector<string>, int> * stats,
    map<vector<string>, int> * big_stats,
    const int threshold) {
  map<vector<string>, int>::iterator it = stats->begin();
  while (it != stats->end()) {
    vector<string> item = it->first;
    int freq = it->second;
    if (freq < threshold) {
      stats->erase(it++);
      if (freq < 0) {
        (*big_stats)[item] += freq;
      }
      else {
        (*big_stats)[item] = freq;
      }
    }
    else {
      ++it;
    }
  }
}

// Make character bigram from the word.
//
// Arguments:
//   word: word vector that is splitted to character level.
//
// Returns:
//   pairs of character bigram
vector<pair<string, string>> getPairs(vector<string> * word) {
  vector<pair<string, string>> pairs;
  string prev_char = (*word)[0];
  for (unsigned i = 1; i < word->size(); i++) {
    pairs.emplace_back(pair<string, string>(prev_char, (*word)[i]));
    prev_char = (*word)[i];
  }
  return pairs;
}

// Encode a word to BPE converted words.
//
// Arguments:
//   orig: original word
//   bpe_codes: bpe_codes made by this class
//   bpe_cache: BPE converted words
// Returns:
//   BPE words
vector<string> encode(const string * orig, 
    const map<pair<string, string>, unsigned> * bpe_codes,
    map<string, vector<string>> * bpe_cache) {
  // if exists in bpe_cache
  const auto &entry = bpe_cache->find(*orig);
  if (entry != bpe_cache->end()) {
    return entry->second;
  }

  vector<string> word = UTF8::getLetters(*orig);
  word.emplace_back("</w>");
  vector<pair<string, string>> pairs = getPairs(&word);

  while (true) {
    unsigned min_bigram = UINT_MAX;
    unsigned argmin_bigram = 0;
    for (unsigned i = 0; i < pairs.size(); i++) {
      const auto &entry = bpe_codes->find(pairs[i]);
      if (entry != bpe_codes->end() and
          entry->second < min_bigram) {
        min_bigram = entry->second;
        argmin_bigram = i;
      }
    }
    if (min_bigram == UINT_MAX) {
      break;
    }
    string first = pairs[argmin_bigram].first;
    string second = pairs[argmin_bigram].second;
    vector<string> new_word;
    unsigned i = 0;
    while (i < word.size()) {
      unsigned j = findIndex(&word, &first, i);
      if (j == word.size()) {
        copy(word.begin() + i, word.end(), back_inserter(new_word));
        break;
      }
      copy(word.begin() + i, word.begin() + j, back_inserter(new_word));
      i = j;

      if (word[i] == first and i < word.size() - 1 and word[i+1] == second) {
        new_word.emplace_back(first + second);
        i += 2;
      } else {
        new_word.emplace_back(word[i]);
        i += 1;
      }
    }
    word = new_word;
    if (word.size() == 1) {
      break;
    } else {
      pairs = getPairs(&word);
    }
  }

  (*bpe_cache)[*orig] = word;
  return word;
}

}  // namespace

namespace nmtkit {

BPEVocabulary::BPEVocabulary(const string & corpus_filename, unsigned size) {
  NMTKIT_CHECK(size >= 3, "Size should be equal or greater than 3.");
  ifstream ifs(corpus_filename);
  NMTKIT_CHECK(
      ifs.is_open(),
      "Could not open corpus file to load: " + corpus_filename);

  // Counts word frequencies.
  map<string, unsigned> char_freq;
  string line;
  unsigned num_lines = 0;
  unsigned num_letters = 0;
  while (Corpus::readLine(&ifs, &line)) {
    ++num_lines;
    line += " ";  // to add </w> at the end of sentence
    vector<string> letters = UTF8::getLetters(line);
    num_letters += letters.size();
    for (string & letter : letters) {
      if (letter == " ") {
        letter = "</w>";
      }
      ++char_freq[letter];
    }
  }
  ifs.clear();
  ifs.seekg(0);

  // Selects most frequent letters.
  vector<pair<unsigned, string>> entries;
  for (const auto & entry : char_freq) {
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

  // begin making BPE codes
  map<vector<string>, int> vocab;
  while (Corpus::readLine(&ifs, &line)) {
    vector<string> words;
    boost::split(
        words, line, boost::is_space(), boost::algorithm::token_compress_on);
    for (const string & word : words) {
      vector<string> key = UTF8::getLetters(word);
      key.emplace_back("</w>");
      ++vocab[key];
    }
  }
  vector<pair<vector<string>, int>> vector_vocab;
  for (auto elm : vocab) {
    vector_vocab.emplace_back(pair<vector<string>, int>(elm.first, elm.second));
  }

  map<vector<string>, int> stats;
  map<vector<string>, map<unsigned, int>> indices;
  getPairStatistics(&vector_vocab, &stats, &indices);
  map<vector<string>, int> big_stats = stats;
  int threshold = stats[findMax(&stats)] / 10;

  unsigned num_letter_vocab = stoi_.size();
  for (unsigned i = 0; i < size - num_letter_vocab; i++) {
    vector<string> most_frequent_index;
    if (!stats.empty()) {
      most_frequent_index = findMax(&stats);
    }
    if (stats.empty() or (i != 0 and stats[most_frequent_index] < threshold)) {
      pruneStats(&stats, &big_stats, threshold);
      stats = big_stats;
      most_frequent_index = findMax(&stats);
      threshold = stats[most_frequent_index] * i/(i+10000.0);
      pruneStats(&stats, &big_stats, threshold);
    }

    // Store entries
    bpe_codes_[pair<string, string>(most_frequent_index[0], most_frequent_index[1])] = i;
    itos_.emplace_back(boost::join(most_frequent_index, ""));
    stoi_[boost::join(most_frequent_index, "")] = i + num_letter_vocab;
    freq_.emplace_back(stats[most_frequent_index]);

    // update vocabulary frequency
    for (const string & word : most_frequent_index) {
      if (stoi_.count(word) != 0) {
        freq_[stoi_[word]] -= stats[most_frequent_index];
      }
    }

    vector<Change> changes =
      replacePair(&most_frequent_index, &vector_vocab, &indices[most_frequent_index]);
    updatePairStatistics(&most_frequent_index, &changes, &stats, &indices);
    stats[most_frequent_index] = 0;

    if (i % 100 == 0) {
      pruneStats(&stats, &big_stats, threshold);
    }
  }
  // end making BPE codes
}

unsigned BPEVocabulary::getID(const string & word) const {
  const auto &entry = stoi_.find(word);
  if (entry == stoi_.end()) return 0;  // ID of <unk>
  return entry->second;
}

string BPEVocabulary::getWord(const unsigned id) const {
  NMTKIT_CHECK(id < itos_.size(), "Index out of range.");
  return itos_[id];
}

unsigned BPEVocabulary::getFrequency(const unsigned id) const {
  NMTKIT_CHECK(id < itos_.size(), "Index out of range.");
  return freq_[id];
}

vector<unsigned> BPEVocabulary::convertToIDs(const string & sentence) const {
  vector<string> words;
  boost::split(
      words, sentence, boost::is_space(), boost::algorithm::token_compress_on);
  vector<unsigned> ids;
  for (const string & word : words) {
    vector<string> new_words = encode(&word, &bpe_codes_, &bpe_cache_);
    for (const string & new_word : new_words) {
      ids.emplace_back(getID(new_word));
    }
  }
  return ids;
}

string BPEVocabulary::convertToSentence(
    const vector<unsigned> & word_ids) const {
  vector<string> words;
  for (const unsigned word_id : word_ids) {
    words.emplace_back(getWord(word_id));
  }
  string sentence = boost::join(words, "");
  boost::replace_all(sentence, "</w>", " ");
  boost::trim_right(sentence);
  return sentence;
}

unsigned BPEVocabulary::size() const {
  return itos_.size();
}

}  // namespace nmtkit

NMTKIT_SERIALIZATION_IMPL(nmtkit::BPEVocabulary);

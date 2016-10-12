#include <nmtkit/corpus.h>

#include <fstream>
#include <boost/algorithm/string.hpp>
#include <nmtkit/exception.h>

using namespace std;

namespace NMTKit {

void Corpus::loadFromTokenFile(
    const string & filepath,
    const NMTKit::Vocabulary & vocab,
    vector<vector<unsigned>> * result) {
  ifstream ifs(filepath);
  NMTKIT_CHECK(
      ifs.is_open(), "Could not open corpus file to load: " + filepath);

  // Loads all lines and converts all words into word IDs.
  result->clear();
  string line;
  while (getline(ifs, line)) {
    boost::trim(line);
    vector<string> words;
    boost::split(
        words, line, boost::is_space(), boost::algorithm::token_compress_on);
    vector<unsigned> word_ids;
    for (const string & word : words) {
      word_ids.emplace_back(vocab.getID(word));
    }
    result->emplace_back(word_ids);
  }
}

}  // namespace NMTKit


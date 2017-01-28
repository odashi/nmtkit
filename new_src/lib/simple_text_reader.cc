#include <nmtkit/simple_text_reader.h>

#include <config.h>
#include <boost/algorithm/string.hpp>
#include <nmtkit/exception.h>
#include <utility>

using std::string;
using std::vector;

namespace nmtkit {

SimpleTextReader::SimpleTextReader(const string & filepath) {
  NMTKIT_CHECK(!filepath.empty());
  ifs_.open(filepath);
  NMTKIT_CHECK_MSG(ifs_.is_open(), "Could not open file to read: " + filepath);
}

SimpleTextReader::~SimpleTextReader() {}

bool SimpleTextReader::read(Sentence * sentence) {
  string line;
  if (!std::getline(ifs_, line)) return false;

  vector<string> words;
  boost::trim(line);
  boost::split(
      words, line, boost::is_space(), boost::algorithm::token_compress_on);
  *sentence = Sentence(words);
  return true;
}

}  // namespace nmtkit

#include "config.h"

#include <nmtkit/simple_text_reader.h>

#include <utility>
#include <boost/algorithm/string.hpp>
#include <nmtkit/exception.h>

using std::string;
using std::vector;

namespace nmtkit {

SimpleTextReader::SimpleTextReader(
    const string & src_filepath,
    const string & trg_filepath) {
  NMTKIT_CHECK(!src_filepath.empty() || !trg_filepath.empty());

  // Loads source file.
  if (!src_filepath.empty()) {
    src_ifs_.open(src_filepath);
    NMTKIT_CHECK_MSG(
        src_ifs_.is_open(),
        "Could not open file to read: " + src_filepath);
  }

  // Loads target file.
  if (!trg_filepath.empty()) {
    trg_ifs_.open(trg_filepath);
    NMTKIT_CHECK_MSG(
        trg_ifs_.is_open(),
        "Could not open file to read: " + trg_filepath);
  }
}

SimpleTextReader::~SimpleTextReader() {}

bool SimpleTextReader::read(SentencePair * sp) {
  vector<string> src_words;
  vector<string> trg_words;

  // Loads one source sentence.
  if (src_ifs_.is_open()) {
    string line;
    if (!std::getline(src_ifs_, line)) return false;
    boost::trim(line);
    boost::split(
        src_words, line,
        boost::is_space(), boost::algorithm::token_compress_on);
  }

  // Loads one target sentence.
  if (trg_ifs_.is_open()) {
    string line;
    if (!std::getline(trg_ifs_, line)) return false;
    boost::trim(line);
    boost::split(
        trg_words, line,
        boost::is_space(), boost::algorithm::token_compress_on);
  }

  // Make one sample.
  SentencePair temp {
    { vector<Token>(src_words.size()), FeatureMap() },
    { vector<Token>(trg_words.size()), FeatureMap() },
    FeatureMap(),
  };
  for (unsigned i = 0; i < src_words.size(); ++i) {
    temp.source.tokens[i].surface = std::move(src_words[i]);
  }
  for (unsigned i = 0; i < trg_words.size(); ++i) {
    temp.target.tokens[i].surface = std::move(trg_words[i]);
  }

  *sp = std::move(temp);
  return true;
}

}  // namespace nmtkit

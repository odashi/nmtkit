#include <nmtkit/monotone_sampler.h>

#include <algorithm>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <nmtkit/exception.h>

using namespace std;

namespace {

void loadCorpus(
    const string & filepath,
    const NMTKit::Vocabulary & vocab,
    vector<vector<int>> * result) {
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
    vector<int> word_ids;
    for (const string & word : words) {
      word_ids.emplace_back(vocab.getID(word));
    }
    result->emplace_back(word_ids);
  }
}

}  // namespace

namespace NMTKit {

MonotoneSampler::MonotoneSampler(
    const string & src_filepath,
    const string & trg_filepath,
    const Vocabulary & src_vocab,
    const Vocabulary & trg_vocab,
    int batch_size,
    bool forever)
: batch_size_(batch_size), forever_(forever) {
  ::loadCorpus(src_filepath, src_vocab, &src_samples_);
  ::loadCorpus(trg_filepath, trg_vocab, &trg_samples_);
  NMTKIT_CHECK_EQ(
      src_samples_.size(), trg_samples_.size(),
      "Number of sentences in source and target corpus are different.");
  NMTKIT_CHECK(batch_size_ > 0, "batch_size should be greater than 0.");
  reset();
}

void MonotoneSampler::rewind() {
  current_ = 0;
}

void MonotoneSampler::reset() {
  rewind();
  iterated_ = 0;
}

void MonotoneSampler::getSamples(vector<Sample> * result) {
  NMTKIT_CHECK(hasSamples(), "No more samples in the sampler.");

  result->clear();
  for (int i = 0; i < batch_size_; ++i) {
    result->emplace_back(
        Sample {src_samples_[current_], trg_samples_[current_]});
    ++current_;
    ++iterated_;
    if (!hasSamples()) {
      if (forever_) {
        rewind();
      } else {
        break;
      }
    }
  }
}

bool MonotoneSampler::hasSamples() const {
  return current_ < src_samples_.size();
}

long MonotoneSampler::numIterated() const {
  return iterated_;
}

}  // namespace NMTKit


#include "config.h"

#include <nmtkit/monotone_sampler.h>

#include <nmtkit/corpus.h>
#include <nmtkit/exception.h>

using std::string;
using std::vector;

namespace nmtkit {

MonotoneSampler::MonotoneSampler(
    const string & src_filepath,
    const string & trg_filepath,
    const Vocabulary & src_vocab,
    const Vocabulary & trg_vocab,
    unsigned max_length,
    float max_length_ratio,
    unsigned batch_size)
: batch_size_(batch_size) {
  Corpus::loadParallelSentences(
      src_filepath, trg_filepath,
      src_vocab, trg_vocab, max_length, max_length_ratio,
      &src_samples_, &trg_samples_);
  NMTKIT_CHECK(src_samples_.size() > 0, "Corpus files are empty.");
  NMTKIT_CHECK(batch_size_ > 0, "batch_size should be greater than 0.");

  rewind();
}

void MonotoneSampler::rewind() {
  current_ = 0;
}

vector<Sample> MonotoneSampler::getSamples() {
  NMTKIT_CHECK(hasSamples(), "No more samples.");

  vector<Sample> result;
  for (unsigned i = 0; i < batch_size_ && hasSamples(); ++i) {
    result.emplace_back(
        Sample {src_samples_[current_], trg_samples_[current_]});
    ++current_;
  }

  return result;
}

unsigned MonotoneSampler::getNumSamples() {
  return src_samples_.size();
}

bool MonotoneSampler::hasSamples() const {
  return current_ < src_samples_.size();
}

}  // namespace nmtkit

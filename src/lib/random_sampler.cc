#include <nmtkit/random_sampler.h>

#include <numeric>
#include <nmtkit/array.h>
#include <nmtkit/corpus.h>
#include <nmtkit/exception.h>

using namespace std;

namespace nmtkit {

RandomSampler::RandomSampler(
    const string & src_filepath,
    const string & trg_filepath,
    const Vocabulary & src_vocab,
    const Vocabulary & trg_vocab,
    unsigned max_length,
    float max_length_ratio,
    unsigned batch_size,
    unsigned random_seed)
: batch_size_(batch_size) {
  Corpus::loadParallelSentences(
      src_filepath, trg_filepath,
      src_vocab, trg_vocab, max_length, max_length_ratio,
      &src_samples_, &trg_samples_);
  NMTKIT_CHECK(src_samples_.size() > 0, "Corpus files are empty.");
  NMTKIT_CHECK(batch_size_ > 0, "batch_size should be greater than 0.");

  ids_.resize(src_samples_.size());
  iota(ids_.begin(), ids_.end(), 0);
  rnd_.reset(random_seed);
  rewind();
}

void RandomSampler::rewind() {
  current_ = 0;
  Array::shuffle(&ids_, &rnd_);
}

void RandomSampler::getSamples(vector<Sample> * result) {
  NMTKIT_CHECK(hasSamples(), "No more samples.");

  result->clear();
  for (unsigned i = 0; i < batch_size_ && hasSamples(); ++i) {
    result->emplace_back(
        Sample {src_samples_[ids_[current_]], trg_samples_[ids_[current_]]});
    ++current_;
  }
}

bool RandomSampler::hasSamples() const {
  return current_ < src_samples_.size();
}

}  // namespace nmtkit

#include <nmtkit/random_sampler.h>

#include <numeric>
#include <nmtkit/corpus.h>
#include <nmtkit/exception.h>

using namespace std;

namespace NMTKit {

RandomSampler::RandomSampler(
    const string & src_filepath,
    const string & trg_filepath,
    const Vocabulary & src_vocab,
    const Vocabulary & trg_vocab,
    int batch_size,
    int random_seed)
: batch_size_(batch_size) {
  Corpus::loadFromTokenFile(src_filepath, src_vocab, &src_samples_);
  Corpus::loadFromTokenFile(trg_filepath, trg_vocab, &trg_samples_);
  NMTKIT_CHECK_EQ(
      src_samples_.size(), trg_samples_.size(),
      "Number of sentences in source and target corpus are different.");
  NMTKIT_CHECK(src_samples_.size() > 0, "Corpus files are empty.");
  NMTKIT_CHECK(batch_size_ > 0, "batch_size should be greater than 0.");

  ids_.resize(src_samples_.size());
  iota(ids_.begin(), ids_.end(), 0);
  rnd_.reset(random_seed);
  rewind();
}

void RandomSampler::rewind() {
  current_ = 0;
  rnd_.shuffle(&ids_);
}

void RandomSampler::getSamples(vector<Sample> * result) {
  NMTKIT_CHECK(hasSamples(), "No more samples.");

  result->clear();
  for (int i = 0; i < batch_size_ && hasSamples(); ++i) {
    result->emplace_back(
        Sample {src_samples_[ids_[current_]], trg_samples_[ids_[current_]]});
    ++current_;
  }
}

bool RandomSampler::hasSamples() const {
  return current_ < src_samples_.size();
}

}  // namespace NMTKit


#include "config.h"

#include <nmtkit/test_sampler.h>

#include <nmtkit/test_corpus.h>
#include <nmtkit/exception.h>

using namespace std;

namespace nmtkit {

TestSampler::TestSampler(
    const string & src_filepath,
    const string & trg_filepath,
    const Vocabulary & src_vocab,
    const Vocabulary & trg_vocab,
    unsigned batch_size)
: batch_size_(batch_size) {
  TestCorpus::loadParallelSentences(
      src_filepath, trg_filepath,
      src_vocab, trg_vocab, 
      &src_samples_, &trg_samples_, &src_samples_string_, &trg_samples_string_);
  NMTKIT_CHECK(src_samples_.size() > 0, "Corpus files are empty.");
  NMTKIT_CHECK(batch_size_ > 0, "batch_size should be greater than 0.");

  rewind();
}

void TestSampler::rewind() {
  current_ = 0;
}

vector<Sample> TestSampler::getSamples() {
  NMTKIT_CHECK(hasSamples(), "No more samples.");

  vector<Sample> result;
  for (unsigned i = 0; i < batch_size_ && hasSamples(); ++i) {
    result.emplace_back(
        Sample {src_samples_[current_], trg_samples_[current_]});
    ++current_;
  }

  return result;
}

vector<TestSample> TestSampler::getTestSamples() {
  NMTKIT_CHECK(hasSamples(), "No more samples.");

  vector<TestSample> result;
  for (unsigned i = 0; i < batch_size_ && hasSamples(); ++i) {
    result.emplace_back(
        TestSample {src_samples_[current_], trg_samples_[current_],
            src_samples_string_[current_], trg_samples_string_[current_]});
    ++current_;
  }

  return result;
}

unsigned TestSampler::getNumSamples() {
  return src_samples_.size();
}

bool TestSampler::hasSamples() const {
  return current_ < src_samples_.size();
}

}  // namespace nmtkit

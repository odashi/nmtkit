#include "config.h"

#include <nmtkit/sorted_random_sampler.h>

#include <algorithm>
#include <nmtkit/array.h>
#include <nmtkit/corpus.h>
#include <nmtkit/exception.h>

using namespace std;

namespace nmtkit {

SortedRandomSampler::SortedRandomSampler(
    const string & src_filepath,
    const string & trg_filepath,
    const Vocabulary & src_vocab,
    const Vocabulary & trg_vocab,
    unsigned max_length,
    float max_length_ratio,
    unsigned num_words_in_batch,
    unsigned random_seed) {
  Corpus::loadParallelSentences(
      src_filepath, trg_filepath,
      src_vocab, trg_vocab, max_length, max_length_ratio,
      &samples_);
  NMTKIT_CHECK(samples_.size() > 0, "Corpus files are empty.");
  NMTKIT_CHECK(
      num_words_in_batch > 0, "num_words_in_batch should be greater than 0.");
  NMTKIT_CHECK(
      num_words_in_batch >= max_length,
      "num_words_in_batch should be greater than or equal to max_length.");

  rnd_.reset(random_seed);

  // Shuffles samples at first to avoid biases due to the original order.
  Array::shuffle(&samples_, &rnd_);

  // Sorts corpus by target lengths.
  Array::sort(&samples_, [](const Sample & a, const Sample & b) {
      return a.target.size() < b.target.size();
  });

  // Searches all batch positions.
  unsigned prev_head = 0;
  unsigned prev_trg_length = 0;
  for (unsigned i = 0; i < samples_.size(); ++i) {
    const Sample & sample = samples_[i];
    unsigned trg_length = max(
        prev_trg_length, static_cast<unsigned>(sample.target.size()));
    // NOTE: Each target outputs in actual batch data has at least one
    //       additional "</s>" tag.
    if ((trg_length + 1) * (i + 1 - prev_head) > num_words_in_batch) {
      positions_.emplace_back(Position {prev_head, i});
      prev_head = i;
    }
    prev_trg_length = trg_length;
  }
  positions_.emplace_back(
      Position {prev_head, static_cast<unsigned>(samples_.size())});

  rewind();
}

void SortedRandomSampler::rewind() {
  current_ = 0;
  Array::shuffle(&positions_, &rnd_);
}

void SortedRandomSampler::getSamples(vector<Sample> * result) {
  NMTKIT_CHECK(hasSamples(), "No more samples.");

  result->clear();
  const Position & pos = positions_[current_];
  for (unsigned i = pos.head; i < pos.tail; ++i) {
    result->emplace_back(samples_[i]);
  }
  ++current_;
}

bool SortedRandomSampler::hasSamples() const {
  return current_ < positions_.size();
}

}  // namespace nmtkit

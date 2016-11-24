#include "config.h"

#include <nmtkit/sorted_random_sampler.h>

#include <algorithm>
#include <tuple>
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
    const string & batch_method,
    const string & sort_method,
    unsigned batch_size,
    unsigned max_length,
    float max_length_ratio,
    unsigned random_seed) {
  Corpus::loadParallelSentences(
      src_filepath, trg_filepath,
      src_vocab, trg_vocab, max_length, max_length_ratio,
      &samples_);
  NMTKIT_CHECK(samples_.size() > 0, "Corpus files are empty.");
  NMTKIT_CHECK(
      batch_size > 0, "batch_size should be greater than 0.");

  rnd_.reset(random_seed);

  // Shuffles samples at first to avoid biases due to the original order.
  Array::shuffle(&samples_, &rnd_);

  // Sorts corpus by selected method.
  if (sort_method == "source") {
    // Use (srclen)
    Array::sort(&samples_, [](const Sample & a, const Sample & b) {
        return a.source.size() < b.source.size();
    });
  } else if (sort_method == "target") {
    // Use (trglen)
    Array::sort(&samples_, [](const Sample & a, const Sample & b) {
        return a.target.size() < b.target.size();
    });
  } else if (sort_method == "source_target") {
    // Use (srclen, trglen)
    Array::sort(&samples_, [](const Sample & a, const Sample & b) {
        return make_tuple(a.source.size(), a.target.size())
            < make_tuple(b.source.size(), b.target.size());
    });
  } else if (sort_method == "target_source") {
    // Use (trglen, srclen)
    Array::sort(&samples_, [](const Sample & a, const Sample & b) {
        return make_tuple(a.target.size(), a.source.size())
            < make_tuple(b.target.size(), b.source.size());
    });
  } else if (sort_method != "none") {
    NMTKIT_FATAL("Invalid name of the sorting strategy: " + sort_method);
  }

  // Searches all batch positions.
  if (batch_method == "target_word") {
    NMTKIT_CHECK(
        batch_size >= max_length,
        "batch_size should be greater than or equal to max_length.");

    unsigned prev_head = 0;
    unsigned prev_len = samples_[0].target.size();
    for (unsigned i = 1; i < samples_.size(); ++i) {
      const unsigned cur_len = samples_[i].target.size();
      const unsigned max_len = max(prev_len, cur_len);
      if (max_len * (i + 1 - prev_head) > batch_size) {
        positions_.emplace_back(Position {prev_head, i});
        prev_head = i;
        prev_len = cur_len;
      } else {
        prev_len = max_len;
      }
    }
    positions_.emplace_back(
        Position {prev_head, static_cast<unsigned>(samples_.size())});
  } else {
    NMTKIT_FATAL("Invalid name of the strategy to make batch: " + batch_method);
  }

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

#include "config.h"

#include <nmtkit/sorted_random_sampler.h>

#include <algorithm>
#include <functional>
#include <tuple>
#include <nmtkit/array.h>
#include <nmtkit/corpus.h>
#include <nmtkit/exception.h>

using namespace std;

namespace {

// Utility function to make all batch positions according to the number of
// words.
//
// Arguments:
//   samples: List of samples.
//   batch_size: Batch size.
//   length_func: Function object which takes a Sample object and returns its
//                length.
//
// Returns:
//   List of batch positions.
vector<nmtkit::SortedRandomSampler::Position> makePositionsByWords(
    const vector<nmtkit::Sample> & samples,
    const unsigned batch_size,
    function<unsigned(const nmtkit::Sample &)> length_func) {
  unsigned prev_head = 0;
  unsigned prev_len = length_func(samples[0]);
  vector<nmtkit::SortedRandomSampler::Position> positions;
  for (unsigned i = 1; i < samples.size(); ++i) {
    const unsigned cur_len = length_func(samples[i]);
    const unsigned max_len = max(prev_len, cur_len);
    if (max_len * (i + 1 - prev_head) > batch_size) {
      positions.emplace_back(
          nmtkit::SortedRandomSampler::Position {prev_head, i});
      prev_head = i;
      prev_len = cur_len;
    } else {
      prev_len = max_len;
    }
  }
  positions.emplace_back(
      nmtkit::SortedRandomSampler::Position {
          prev_head, static_cast<unsigned>(samples.size())});
  return positions;
}

}  // namespace

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
  if (batch_method == "sentence") {
    const unsigned num_samples = samples_.size();
    for (unsigned i = 0; i < num_samples; i += batch_size) {
      positions_.emplace_back(Position {i, min(i + batch_size, num_samples)});
    }
  } else if (batch_method == "both_word") {
    NMTKIT_CHECK(
        batch_size >= 2 * max_length,
        "If batch_method == \"both_word\", "
        "batch_size should be greater than or equal to (2 * max_length).");
    positions_ = ::makePositionsByWords(
        samples_, batch_size,
        [](const Sample & s) { return s.source.size() + s.target.size(); });
  } else if (batch_method == "source_word") {
    NMTKIT_CHECK(
        batch_size >= max_length,
        "If batch_method == \"source_word\", "
        "batch_size should be greater than or equal to max_length.");
    positions_ = ::makePositionsByWords(
        samples_, batch_size,
        [](const Sample & s) { return s.source.size(); });
  } else if (batch_method == "target_word") {
    NMTKIT_CHECK(
        batch_size >= max_length,
        "If batch_method == \"target_word\", "
        "batch_size should be greater than or equal to max_length.");
    positions_ = ::makePositionsByWords(
        samples_, batch_size,
        [](const Sample & s) { return s.target.size(); });
  } else {
    NMTKIT_FATAL("Invalid name of the strategy to make batch: " + batch_method);
  }

  rewind();
}

void SortedRandomSampler::rewind() {
  current_ = 0;
  Array::shuffle(&positions_, &rnd_);
}

vector<Sample> SortedRandomSampler::getSamples() {
  NMTKIT_CHECK(hasSamples(), "No more samples.");

  vector<Sample> result;
  const Position & pos = positions_[current_];
  for (unsigned i = pos.head; i < pos.tail; ++i) {
    result.emplace_back(samples_[i]);
  }
  ++current_;

  return result;
}

int SortedRandomSampler::getCorpusSize() {
  return samples_.size();
}

bool SortedRandomSampler::hasSamples() const {
  return current_ < positions_.size();
}

}  // namespace nmtkit

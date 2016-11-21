#ifndef NMTKIT_SORTED_RANDOM_SAMPLER_H_
#define NMTKIT_SORTED_RANDOM_SAMPLER_H_

#include <nmtkit/random.h>
#include <nmtkit/sampler.h>
#include <nmtkit/vocabulary.h>

namespace nmtkit {

class SortedRandomSampler : public Sampler {
  SortedRandomSampler() = delete;
  SortedRandomSampler(const SortedRandomSampler &) = delete;
  SortedRandomSampler(SortedRandomSampler &&) = delete;
  SortedRandomSampler & operator=(const SortedRandomSampler &) = delete;
  SortedRandomSampler & operator=(SortedRandomSampler &&) = delete;

public:
  // Creates sampler.
  //
  // Arguments:
  //   src_filepath: Location of the source corpus.
  //   trg_filepath: Location of the target corpus.
  //   src_vocab: Vocabulary object for the source language.
  //   trg_vocab: Vocabulary object for the target language.
  //   batch_method: Name of the strategy to make batch.
  //                 Available values:
  //                   "target_word" : Make batch data according to the number
  //                                   of target words.
  //   sort_method: Name of the strategy to sort source/target corpus.
  //                Available values:
  //                  "target_source" : First sort by target lengths, then sort
  //                                    by source lengths with maintaining the
  //                                    order of target lengths.
  //   batch_size: Batch size. The meaning of this argument is determined by
  //   max_length: Maximum number of words in a sentence.
  //   max_length_ratio: Maximum ratio of lengths in source/target sentence.
  //               the value of `batch_method`.
  //   random_seed: Seed value of the randomizer to be used for shuffling.
  SortedRandomSampler(
      const std::string & src_filepath,
      const std::string & trg_filepath,
      const Vocabulary & src_vocab,
      const Vocabulary & trg_vocab,
      const std::string & batch_method,
      const std::string & sort_method,
      unsigned batch_size,
      unsigned max_length,
      float max_length_ratio,
      unsigned random_seed);

  ~SortedRandomSampler() override {}

  void rewind() override;
  void getSamples(std::vector<Sample> * result) override;
  bool hasSamples() const override;

private:
  struct Position {
    unsigned head;
    unsigned tail;
  };

  std::vector<Sample> samples_;

  Random rnd_;
  std::vector<Position> positions_;
  unsigned current_;
};

}  // namespace nmtkit

#endif  // NMTKIT_SORTED_RANDOM_SAMPLER_H_

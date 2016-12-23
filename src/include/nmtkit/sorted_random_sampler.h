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
  // Range of each batch data in the internal array.
  struct Position {
    unsigned head;
    unsigned tail;
  };

  // Creates sampler.
  //
  // Arguments:
  //   src_filepath: Location of the source corpus.
  //   trg_filepath: Location of the target corpus.
  //   src_vocab: Vocabulary object for the source language.
  //   trg_vocab: Vocabulary object for the target language.
  //   batch_method: Name of the strategy to make batches.
  //                 Available values:
  //                   "sentence" : According to the number of sentences.
  //                   "both_word" : Accotding to the number of source and
  //                                 target words.
  //                   "source_word" : Accotding to the number of source words.
  //                   "target_word" : According to the number of target words.
  //   sort_method: Name of the strategy to sort source/target corpus.
  //                Available values:
  //                  "none" : Never sort the corpus.
  //                  "source" : Sort by source lengths.
  //                  "target" : Sort by target lengths.
  //                  "source_target" : First sort by source lengths, then sort
  //                                    by target lengths with maintaining the
  //                                    order of source lengths.
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
  std::vector<Sample> getSamples() override;
  int getCorpusSize() override;
  bool hasSamples() const override;

private:
  std::vector<Sample> samples_;

  Random rnd_;
  std::vector<Position> positions_;
  unsigned current_;
};

}  // namespace nmtkit

#endif  // NMTKIT_SORTED_RANDOM_SAMPLER_H_

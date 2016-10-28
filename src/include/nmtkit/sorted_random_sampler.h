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
  SortedRandomSampler(
      const std::string & src_filepath,
      const std::string & trg_filepath,
      const Vocabulary & src_vocab,
      const Vocabulary & trg_vocab,
      unsigned max_length,
      float max_length_ratio,
      unsigned num_words_in_batch,
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

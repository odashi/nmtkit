#ifndef NMTKIT_RANDOM_SAMPLER_H_
#define NMTKIT_RANDOM_SAMPLER_H_

#include <nmtkit/random.h>
#include <nmtkit/sampler.h>
#include <nmtkit/vocabulary.h>

namespace nmtkit {

class RandomSampler : public Sampler {
  RandomSampler() = delete;
  RandomSampler(const RandomSampler &) = delete;
  RandomSampler(RandomSampler &&) = delete;
  RandomSampler & operator=(const RandomSampler &) = delete;
  RandomSampler & operator=(RandomSampler &&) = delete;

public:
  RandomSampler(
      const std::string & src_filepath,
      const std::string & trg_filepath,
      const Vocabulary & src_vocab,
      const Vocabulary & trg_vocab,
      unsigned max_length,
      unsigned batch_size,
      unsigned random_seed);

  ~RandomSampler() override {}

  void rewind() override;
  void getSamples(std::vector<Sample> * result) override;
  bool hasSamples() const override;

private:
  std::vector<std::vector<unsigned>> src_samples_;
  std::vector<std::vector<unsigned>> trg_samples_;
  unsigned batch_size_;
  unsigned current_;
  
  Random rnd_;
  std::vector<unsigned> ids_;
};

}  // namespace nmtkit

#endif  // NMTKIT_RANDOM_SAMPLER_H_

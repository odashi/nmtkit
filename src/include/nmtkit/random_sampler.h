#ifndef NMTKIT_RANDOM_SAMPLER_H_
#define NMTKIT_RANDOM_SAMPLER_H_

#include <nmtkit/random.h>
#include <nmtkit/sampler.h>
#include <nmtkit/vocabulary.h>

namespace NMTKit {

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
      int batch_size,
      int random_seed);

  ~RandomSampler() override {}

  void rewind() override;
  void getSamples(std::vector<Sample> * result) override;
  bool hasSamples() const override;

private:
  std::vector<std::vector<int>> src_samples_;
  std::vector<std::vector<int>> trg_samples_;
  int batch_size_;
  int current_;
  
  Random rnd_;
  std::vector<int> ids_;
};

}  // namespace NMTKit

#endif  // NMTKIT_RANDOM_SAMPLER_H_


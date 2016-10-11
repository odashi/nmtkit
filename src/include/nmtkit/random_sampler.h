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
      bool forever,
      int random_seed);

  ~RandomSampler() override {}

  void reset() override;
  void getSamples(std::vector<Sample> * result) override;
  bool hasSamples() const override;
  long numIterated() const override;

private:
  void rewind();

  std::vector<std::vector<int>> src_samples_;
  std::vector<std::vector<int>> trg_samples_;
  int batch_size_;
  bool forever_;
  int current_;
  long iterated_;
  
  Random rnd_;
  std::vector<int> ids_;
  int random_seed_;
};

}  // namespace NMTKit

#endif  // NMTKIT_RANDOM_SAMPLER_H_


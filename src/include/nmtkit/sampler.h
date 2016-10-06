#ifndef NMTKIT_SAMPLER_H_
#define NMTKIT_SAMPLER_H_

#include <string>
#include <vector>
#include <nmtkit/vocabulary.h>

namespace NMTKit {

struct Sample {
  std::vector<int> source;
  std::vector<int> target;
};

class Sampler {
  public:
    Sampler() {}
    virtual ~Sampler() {}

    virtual void reset() = 0;

    virtual void getSamples(std::vector<Sample> *result) = 0;
    virtual bool hasSamples() const = 0;
};

class MonotoneSampler : public Sampler {
  public:
    MonotoneSampler(
        const std::string &src_filepath, const std::string &trg_filepath,
        const Vocabulary &src_vocab, const Vocabulary &trg_vocab);
    virtual ~MonotoneSampler() override {}

    virtual void reset() override;

    virtual void getSamples(std::vector<Sample> *result) override;
    virtual bool hasSamples() const override;
  
  private:
    std::vector<std::vector<int>> src_samples_;
    std::vector<std::vector<int>> trg_samples_;
    int current_;
};

}  // namespace NMTKit

#endif  // NMTKIT_SAMPLER_H_


#ifndef NMTKIT_MONOTONE_SAMPLER_H_
#define NMTKIT_MONOTONE_SAMPLER_H_

#include <string>
#include <vector>
#include <nmtkit/sampler.h>
#include <nmtkit/vocabulary.h>

namespace NMTKit {

class MonotoneSampler : public Sampler {
  MonotoneSampler() = delete;
  MonotoneSampler(const MonotoneSampler &) = delete;
  MonotoneSampler(MonotoneSampler &&) = delete;
  MonotoneSampler & operator=(const MonotoneSampler &) = delete;
  MonotoneSampler & operator=(MonotoneSampler &&) = delete;

  public:
    MonotoneSampler(
        const std::string & src_filepath,
        const std::string & trg_filepath,
        const Vocabulary & src_vocab,
        const Vocabulary & trg_vocab,
        int batch_size,
        bool forever);

    ~MonotoneSampler() override {}

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
};

}  // namespace NMTKit

#endif  // NMTKIT_MONOTONE_SAMPLER_H_


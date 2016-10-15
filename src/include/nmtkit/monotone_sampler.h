#ifndef NMTKIT_MONOTONE_SAMPLER_H_
#define NMTKIT_MONOTONE_SAMPLER_H_

#include <nmtkit/sampler.h>
#include <nmtkit/vocabulary.h>

namespace nmtkit {

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
      unsigned max_length,
      unsigned batch_size);

  ~MonotoneSampler() override {}

  void rewind() override;
  void getSamples(std::vector<Sample> * result) override;
  bool hasSamples() const override;

private:
  std::vector<std::vector<unsigned>> src_samples_;
  std::vector<std::vector<unsigned>> trg_samples_;
  unsigned batch_size_;
  unsigned current_;
};

}  // namespace nmtkit

#endif  // NMTKIT_MONOTONE_SAMPLER_H_


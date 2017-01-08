#ifndef NMTKIT_TEST_SAMPLER_H_
#define NMTKIT_TEST_SAMPLER_H_

#include <nmtkit/sampler.h>
#include <nmtkit/vocabulary.h>

namespace nmtkit {

class TestSampler : public Sampler {
  TestSampler() = delete;
  TestSampler(const TestSampler &) = delete;
  TestSampler(TestSampler &&) = delete;
  TestSampler & operator=(const TestSampler &) = delete;
  TestSampler & operator=(TestSampler &&) = delete;

public:
  TestSampler(
      const std::string & src_filepath,
      const std::string & trg_filepath,
      const Vocabulary & src_vocab,
      const Vocabulary & trg_vocab,
      unsigned batch_size);

  ~TestSampler() override {}

  void rewind() override;
  std::vector<Sample> getSamples() override;
  std::vector<TestSample> getTestSamples();
  unsigned getNumSamples() override;
  bool hasSamples() const override;

private:
  std::vector<std::string> src_samples_string_;
  std::vector<std::string> trg_samples_string_;
  std::vector<std::vector<unsigned>> src_samples_;
  std::vector<std::vector<unsigned>> trg_samples_;
  unsigned batch_size_;
  unsigned current_;
};

}  // namespace nmtkit

#endif  // NMTKIT_TEST_SAMPLER_H_

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
  Sampler() = delete;
  Sampler(const Sampler &) = delete;
  Sampler(Sampler &&) = delete;
  Sampler & operator=(const Sampler &) = delete;
  Sampler & operator=(Sampler &&) = delete;

  public:
    Sampler(
        const std::string & src_filepath,
        const std::string & trg_filepath,
        const Vocabulary & src_vocab,
        const Vocabulary & trg_vocab,
        int batch_size,
        bool forever);

    void reset();

    void getSamples(std::vector<Sample> * result);

    bool hasSamples() const;
    long numIterated() const;
  
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

#endif  // NMTKIT_SAMPLER_H_


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
    Sampler(
        const std::string &src_filepath,
        const std::string &trg_filepath,
        const Vocabulary &src_vocab, 
        const Vocabulary &trg_vocab,
        bool forever);

    void reset();

    void getSamples(std::vector<Sample> *result);
    bool hasSamples() const;
  
  private:
    std::vector<std::vector<int>> src_samples_;
    std::vector<std::vector<int>> trg_samples_;
    int current_;
    bool forever_;
};

}  // namespace NMTKit

#endif  // NMTKIT_SAMPLER_H_


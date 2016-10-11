#ifndef NMTKIT_SAMPLER_H_
#define NMTKIT_SAMPLER_H_

#include <string>
#include <vector>
#include <nmtkit/basic_types.h>

namespace NMTKit {

// Abstract class to define the interface of sample iteration.
class Sampler {
  Sampler(const Sampler &) = delete;
  Sampler(Sampler &&) = delete;
  Sampler & operator=(const Sampler &) = delete;
  Sampler & operator=(Sampler &&) = delete;

public:
  Sampler() {}
  virtual ~Sampler() {}

  // Reset all inner states. Sampler should starts iterating samples with always
  // same order after resetting.
  virtual void reset() = 0;

  // Retrieves next samples.
  // Arguments:
  //   result: Placeholder to store new samples. Old data will be deleted
  //           automatically before storing new samples.
  virtual void getSamples(std::vector<Sample> * result) = 0;

  // Checks whether or not the sampler has unprocessed samples.
  // Returns:
  //   true if the sampler has more samples, false otherwise.
  virtual bool hasSamples() const = 0;

  // Retrieves number of already iterated samples.
  // After calling reset(), this value sill be set as 0.
  virtual long numIterated() const = 0;
};

}  // namespace NMTKit

#endif  // NMTKIT_SAMPLER_H_


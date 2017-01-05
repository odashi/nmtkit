#ifndef NMTKIT_SAMPLER_H_
#define NMTKIT_SAMPLER_H_

#include <string>
#include <vector>
#include <nmtkit/basic_types.h>

namespace nmtkit {

// Abstract class to define the interface of sample iteration.
class Sampler {
  Sampler(const Sampler &) = delete;
  Sampler(Sampler &&) = delete;
  Sampler & operator=(const Sampler &) = delete;
  Sampler & operator=(Sampler &&) = delete;

public:
  Sampler() {}
  virtual ~Sampler() {}

  // Rewinds input sequence.
  virtual void rewind() = 0;

  // Retrieves next samples.
  // Returns:
  //   List of new samples.
  virtual std::vector<Sample> getSamples() = 0;

  // Rerieves the number of filtered samples.
  // Returns:
  //   Number of filtered samples.
  virtual int getNumSamples() = 0;

  // Checks whether or not the sampler has unprocessed samples.
  // Returns:
  //   true if the sampler has more samples, false otherwise.
  virtual bool hasSamples() const = 0;
};

}  // namespace nmtkit

#endif  // NMTKIT_SAMPLER_H_

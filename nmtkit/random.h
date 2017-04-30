#ifndef NMTKIT_RANDOM_H_
#define NMTKIT_RANDOM_H_

#include <algorithm>
#include <random>
#include <vector>

namespace nmtkit {

// Customized random sampler
class Random {
  Random(const Random &) = delete;
  Random(Random &&) = delete;
  Random & operator=(const Random &) = delete;
  Random & operator=(Random &&) = delete;

public:
  Random();

  // Initializes all states.
  // Arguments:
  //   seed: Seed value for the internal randomizer.
  void reset(unsigned seed);

  // Generates an integer with range [minval, maxval) by sampling from the
  // uniform distribution.
  // Arguments:
  //   minval: Lower bound of generated values. This value could be generated.
  //   maxval: Upper bound of generated values. This value could not be
  //           generated. Actual maximum value is (maxval - 1).
  // Returns:
  //   Generated integer value.
  int uniform(int minval, int maxval);

  // Generates an float with range [minval, maxval) by sampling from the
  // uniform distribution.
  // Arguments:
  //   minval: Lower bound of generated values.
  //   maxval: Upper bound of generated values.
  // Returns:
  //   Generated double value.
  double funiform(double minval, double maxval);

private:
  std::mt19937 gen_;
};

}  // namespace nmtkit

#endif  // NMTKIT_RANDOM_H_

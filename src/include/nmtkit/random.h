#ifndef NMTKIT_RANDOM_H_
#define NMTKIT_RANDOM_H_

#include <random>
#include <vector>

namespace NMTKit {

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
  void reset(unsigned int seed);

  // Generates an integer with range [minval, maxval) by sampling from the
  // uniform distribution.
  // Arguments:
  //   minval: Lower bound of generated values. This value could be generated.
  //   maxval: Upper bound of generated values. This value could not be
  //           generated. Actual maximum value is (maxval - 1).
  // Returns:
  //   Generated integer value.
  int uniform(int minval, int maxval);

  // Shuffles given integer vector.
  // Arguments:
  //   arr: Target vector.
  void shuffle(std::vector<int> * arr);

private:
  std::mt19937 gen_;
};

}  // namespace NMTKit

#endif  // NMTKIT_RANDOM_H_


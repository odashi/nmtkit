#include <nmtkit/random.h>

#include <algorithm>
#include <nmtkit/exception.h>

using namespace std;

namespace NMTKit {

Random::Random() : gen_() {
  NMTKIT_CHECK_EQ(
      0, gen_.min(), "Minimum value of the random generator should be 0.");
  NMTKIT_CHECK_EQ(
      0xffffffff, gen_.max(),
      "Maximum value of the random generator should be 0xffffffff.");
  reset(0);  // default seed
}

void Random::reset(unsigned int seed) {
  gen_.seed(seed);
}

int Random::uniform(int minval, int maxval) {
  NMTKIT_CHECK(minval < maxval, "Arguments should satisfy minval < maxval.");

  // NOTE: This calculation always rejects 0xffffffff from sampled values.
  const unsigned int span = maxval - minval;
  const unsigned int divisor = 0xffffffff / span;
  const unsigned int border = span * divisor;
  unsigned int sample = gen_();
  while (sample >= border) {
    sample = gen_();
  }
  return minval + static_cast<int>(sample / divisor);
}

void Random::shuffle(vector<int> * arr) {
  // Implementing Fisher-Yates algorithm.
  const int M = arr->size();
  for (int i = 0; i < M - 1; ++i) {
    const int j = uniform(i, M);
    if (j > i) {
      swap((*arr)[i], (*arr)[j]);
    }
  }
}

}  // namespace NMTKit


#include <config.h>

#include <nmtkit/random.h>

#include <algorithm>
#include <nmtkit/exception.h>

namespace nmtkit {

Random::Random() : gen_() {
  NMTKIT_CHECK_EQ(
      0, gen_.min(), "Minimum value of the random generator should be 0.");
  NMTKIT_CHECK_EQ(
      0xffffffff, gen_.max(),
      "Maximum value of the random generator should be 0xffffffff.");
  reset(0);  // default seed
}

void Random::reset(unsigned seed) {
  if (seed == 0) {
    std::random_device rd;
    gen_.seed(rd());
  } else {
    gen_.seed(seed);
  }
}

int Random::uniform(int minval, int maxval) {
  NMTKIT_CHECK(minval < maxval, "Arguments should satisfy minval < maxval.");

  // NOTE: This calculation always rejects 0xffffffff from sampled values.
  const unsigned span = maxval - minval;
  const unsigned divisor = 0xffffffff / span;
  const unsigned border = span * divisor;
  unsigned sample = gen_();
  while (sample >= border) {
    sample = gen_();
  }
  return minval + static_cast<int>(sample / divisor);
}

// TODO(odashi): Make unit tests for this function.
double Random::funiform(double minval, double maxval) {
  NMTKIT_CHECK(minval < maxval, "Arguments should satisfy minval < maxval.");

  static const double divisor = static_cast<double>(0x100000000);
  return gen_() * (maxval - minval) / divisor + minval;
}

}  // namespace nmtkit

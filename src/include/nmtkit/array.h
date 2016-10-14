#ifndef NMTKIT_ARRAY_H_
#define NMTKIT_ARRAY_H_

#include <nmtkit/random.h>

namespace NMTKit {

// Array manipulators.
class Array {
public:
  // Shuffles given vector.
  // Arguments:
  //   arr: Target vector.
  //   rnd: Random object to be used.
  template<typename T>
  static void shuffle(std::vector<T> * arr, Random * rnd) {
    // Implementing Fisher-Yates algorithm.
    const unsigned M = arr->size();
    for (unsigned i = 0; i < M - 1; ++i) {
      const unsigned j = rnd->uniform(i, M);
      if (j > i) {
        std::swap((*arr)[i], (*arr)[j]);
      }
    }
  }
};

}  // namespace NMTKit

#endif  // NMTKIT_ARRAY_H_


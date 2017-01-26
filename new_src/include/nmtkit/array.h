#ifndef NMTKIT_ARRAY_H_
#define NMTKIT_ARRAY_H_

#include <algorithm>
#include <functional>
#include <numeric>
#include <utility>
#include <nmtkit/exception.h>
#include <nmtkit/random.h>

namespace nmtkit {

// Array manipulators.
class Array {
public:
  // Reverse given vector.
  // Arguments:
  // arr: Target vector.
  template <typename T>
  static void reverse(std::vector<T> * arr) {
    std::reverse(arr->begin(), arr->end());
  }

  // Sort given vector using less (<) function.
  // Arguments:
  //   arr: Target vector.
  template <typename T>
  static void sort(std::vector<T> * arr) {
    sort(arr, std::less<T>());
  }

  // Sort given vector.
  // Arguments:
  //   arr: Target vector.
  //   less: Predicate indicating first arg < second arg.
  template <typename T, typename P>
  static void sort(std::vector<T> * arr, P less) {
    // Implementing heap sort.
    auto downheap = [&](int k, int r) {
      while (true) {
        int j = (k << 1) + 1;
        if (j > r) break;
        if (j != r && less((*arr)[j], (*arr)[j + 1])) ++j;
        if (less((*arr)[j], (*arr)[k])) break;
        std::swap((*arr)[k], (*arr)[j]);
        k = j;
      }
    };
    const int n = arr->size();
    for (int i = (n - 2) >> 1; i >= 0; --i) {
      downheap(i, n - 1);
    }
    for (int i = n - 1; i > 0; --i) {
      std::swap((*arr)[0], (*arr)[i]);
      downheap(0, i - 1);
    }
  }

  // Shuffles given vector.
  // Arguments:
  //   arr: Target vector.
  //   rnd: Random object to be used.
  template <typename T>
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

  // Retrieves argmax value of given vector using greater(>) function.
  // Arguments:
  //   arr: Target vector.
  //
  // Returns:
  //   Argmax index value.
  template <typename T>
  static unsigned argmax(const std::vector<T> & arr) {
    return argmax(arr, std::greater<T>());
  }

  // Retrieves argmax value of given vector.
  // Arguments:
  //   arr: Target vector.
  //   greater: Predicate indicating first-arg > second-arg.
  //
  // Returns:
  //   Argmax index value.
  template <typename T, typename P>
  static unsigned argmax(const std::vector<T> & arr, P greater) {
    NMTKIT_CHECK(arr.size() > 0);
    unsigned ret = 0;
    for (unsigned i = 1; i < arr.size(); ++i) {
      if (greater(arr[i], arr[ret])) {
        ret = i;
      }
    }
    return ret;
  }

  // Retrieves k-best indices of given vector using greater(>) function.
  // Arguments:
  //   arr: Target vector.
  //   num_results: Number of results to be obtained.
  // Returns:
  //   K-best indices.
  template <typename T>
  static std::vector<unsigned> kbest(
      const std::vector<T> & arr,
      const unsigned num_results) {
    return kbest(arr, num_results, std::greater<T>());
  }

  // Retrieves k-best indices of given vector.
  // Arguments:
  //   arr: Target vector.
  //   num_results: Number of results to be obtained.
  //   greater: Predicate indicating first-arg > second-arg.
  // Returns:
  //   K-best indices.
  template <typename T, typename P>
  static std::vector<unsigned> kbest(
      const std::vector<T> & arr,
      const unsigned num_results,
      P greater) {
    // Implementing based on heap sort
    const int n = arr.size();
    NMTKIT_CHECK(n >= static_cast<int>(num_results));
    std::vector<unsigned> ids(n);
    std::iota(ids.begin(), ids.end(), 0);
    auto downheap = [&](int k, int r) {
      while (true) {
        int j = (k << 1) + 1;
        if (j > r) break;
        if (j != r && greater(arr[ids[j + 1]], arr[ids[j]])) ++j;
        if (greater(arr[ids[k]], arr[ids[j]])) break;
        std::swap(ids[k], ids[j]);
        k = j;
      }
    };
    for (int i = (n - 2) >> 1; i >= 0; --i) {
      downheap(i, n - 1);
    }
    const int border = n - num_results;
    for (int i = n - 1; i > 0 && i >= border; --i) {
      std::swap(ids[0], ids[i]);
      downheap(0, i - 1);
    }
    std::vector<unsigned> results;
    for (int i = n - 1; i >= border; --i) {
      results.emplace_back(ids[i]);
    }
    return results;
  }
};

}  // namespace nmtkit

#endif  // NMTKIT_ARRAY_H_

#ifndef NMTKIT_BASIC_TYPES_H_
#define NMTKIT_BASIC_TYPES_H_

#include <string>
#include <vector>

namespace nmtkit {

struct Sample {
  std::vector<unsigned> source;
  std::vector<unsigned> target;
};

}  // namespace nmtkit

#endif  // NMTKIT_BASIC_TYPES_H_


#ifndef NMTKIT_BASIC_TYPES_H_
#define NMTKIT_BASIC_TYPES_H_

#include <string>
#include <vector>

namespace NMTKit {

struct Sample {
  std::vector<int> source;
  std::vector<int> target;
};

}  // namespace NMTKit

#endif  // NMTKIT_BASIC_TYPES_H_


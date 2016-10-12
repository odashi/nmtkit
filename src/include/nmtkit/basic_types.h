#ifndef NMTKIT_BASIC_TYPES_H_
#define NMTKIT_BASIC_TYPES_H_

#include <string>
#include <vector>

namespace NMTKit {

struct Sample {
  std::vector<unsigned> source;
  std::vector<unsigned> target;
};

}  // namespace NMTKit

#endif  // NMTKIT_BASIC_TYPES_H_


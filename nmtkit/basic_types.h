#ifndef NMTKIT_BASIC_TYPES_H_
#define NMTKIT_BASIC_TYPES_H_

#include <string>
#include <vector>

namespace nmtkit {

struct Sample {
  // Source sentence with word IDs
  std::vector<unsigned> source;

  // Target sentence with word IDs.
  std::vector<unsigned> target;
};

struct Batch {
  // Source word ID table with shape (max_source_length, batch_size).
  std::vector<std::vector<unsigned>> source_ids;
  
  // Target word iID table with shape (max_source_length, batch_size).
  std::vector<std::vector<unsigned>> target_ids;
};

}  // namespace nmtkit

#endif  // NMTKIT_BASIC_TYPES_H_

#ifndef NMTKIT_BATCH_CONVERTER_H_
#define NMTKIT_BATCH_CONVERTER_H_

#include <vector>
#include <nmtkit/basic_types.h>

namespace nmtkit {

class BatchConverter {
public:
  // Converts Sample into Batch.
  // Arguments:
  //   samples: Input Sample vector.
  //   eos_id: End-of-sentence word ID.
  //   batch: Output Batch object.
  static void convert(
      const std::vector<Sample> & samples,
      unsigned eos_id,
      Batch * batch);
};

}  // namespace nmtkit

#endif  // NMTKIT_BATCH_CONVERTER_H_


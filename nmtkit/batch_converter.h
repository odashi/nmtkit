#ifndef NMTKIT_BATCH_CONVERTER_H_
#define NMTKIT_BATCH_CONVERTER_H_

#include <vector>
#include <nmtkit/basic_types.h>
#include <nmtkit/vocabulary.h>

namespace nmtkit {

class BatchConverter {
  BatchConverter() = delete;
  BatchConverter(const BatchConverter &) = delete;
  BatchConverter(BatchConverter &&) = delete;
  BatchConverter & operator=(const BatchConverter &) = delete;
  BatchConverter & operator=(BatchConverter &&) = delete;

public:
  // Make new BatchConverter object.
  // Arguments:
  //   src_vocab: Vocabulary object for the source language.
  //   trg_vocab: Vocabulary object for the target language.
  BatchConverter(const Vocabulary & src_vocab, const Vocabulary & trg_vocab);

  // Converts Sample into Batch.
  // Arguments:
  //   samples: Input Sample vector.
  //
  // Returns:
  //   Output Batch object.
  Batch convert(const std::vector<Sample> & samples);

private:
  unsigned src_bos_id_;
  unsigned src_eos_id_;
  unsigned trg_bos_id_;
  unsigned trg_eos_id_;
};

}  // namespace nmtkit

#endif  // NMTKIT_BATCH_CONVERTER_H_

#ifndef NMTKIT_BLEU_EVALUATOR_H_
#define NMTKIT_BLEU_EVALUATOR_H_

#include <nmtkit/evaluator.h>

#include <string>
#include <nmtkit/monotone_sampler.h>
#include <nmtkit/vocabulary.h>

namespace nmtkit {

// Evaluator class to calculate corpus-wise BLEU score.
class BLEUEvaluator : public Evaluator {
  BLEUEvaluator(const BLEUEvaluator &) = delete;
  BLEUEvaluator(BLEUEvaluator &&) = delete;
  BLEUEvaluator & operator=(const BLEUEvaluator &) = delete;
  BLEUEvaluator & operator=(BLEUEvaluator &&) = delete;

public:
  // Constructs an evaluator.
  //
  // Arguments:
  //   src_filepath: Location of the source corpus.
  //   trg_filepath: Location of the target corpus.
  //   src_vocab: Pointer of the soruce vocabulary.
  //   trg_vocab: Pointer of the target vocabulary.
  BLEUEvaluator(
      const std::string & src_filepath,
      const std::string & trg_filepath,
      const Vocabulary & src_vocab,
      const Vocabulary & trg_vocab);
  ~BLEUEvaluator() override {}

  float evaluate(EncoderDecoder * encdec) override;

private:
  MonotoneSampler sampler_;
  unsigned src_bos_id_;
  unsigned src_eos_id_;
  unsigned trg_bos_id_;
  unsigned trg_eos_id_;
};

} // namespace nmtkit

#endif // NMTKIT_BLEU_EVALUATOR_H_

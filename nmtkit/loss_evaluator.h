#ifndef NMTKIT_LOSS_EVALUATOR_H_
#define NMTKIT_LOSS_EVALUATOR_H_

#include <nmtkit/evaluator.h>

#include <string>
#include <nmtkit/batch_converter.h>
#include <nmtkit/monotone_sampler.h>
#include <nmtkit/vocabulary.h>

namespace nmtkit {

// Evaluator class to calculate averaged loss function.
class LossEvaluator : public Evaluator {
  LossEvaluator(const LossEvaluator &) = delete;
  LossEvaluator(LossEvaluator &&) = delete;
  LossEvaluator & operator=(const LossEvaluator &) = delete;
  LossEvaluator & operator=(LossEvaluator &&) = delete;

public:
  // Constructs an evaluator.
  //
  // Arguments:
  //   src_filepath: Location of the source corpus.
  //   trg_filepath: Location of the target corpus.
  //   src_vocab: Pointer of the soruce vocabulary.
  //   trg_vocab: Pointer of the target vocabulary.
  LossEvaluator(
      const std::string & src_filepath,
      const std::string & trg_filepath,
      const Vocabulary & src_vocab,
      const Vocabulary & trg_vocab);
  ~LossEvaluator() override {}

  float evaluate(EncoderDecoder * encdec) override;

private:
  BatchConverter converter_;
  MonotoneSampler sampler_;
};

} // namespace nmtkit

#endif // NMTKIT_LOSS_EVALUATOR_H_

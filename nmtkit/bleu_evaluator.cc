#include <nmtkit/bleu_evaluator.h>

#include <vector>
#include <mteval/EvaluatorFactory.h>
#include <nmtkit/exception.h>

using std::string;
using std::vector;

namespace {
  const unsigned MAX_SEQUENCE_LENGTH = 1024;
  const float MAX_SEQUENCE_RATIO = 1024.0f;
  const unsigned BATCH_SIZE = 1;
  const unsigned BEAM_WIDTH = 1;
  const float WORD_PENALTY = 0.0f;
} // namespace

namespace nmtkit {

BLEUEvaluator::BLEUEvaluator(
    const string & src_filepath,
    const string & trg_filepath,
    const Vocabulary & src_vocab,
    const Vocabulary & trg_vocab)
: sampler_(
    src_filepath, trg_filepath,
    src_vocab, trg_vocab,
    ::MAX_SEQUENCE_LENGTH, ::MAX_SEQUENCE_RATIO, ::BATCH_SIZE)
, src_bos_id_(src_vocab.getID("<s>"))
, src_eos_id_(src_vocab.getID("</s>"))
, trg_bos_id_(trg_vocab.getID("<s>"))
, trg_eos_id_(trg_vocab.getID("</s>"))
{}

float BLEUEvaluator::evaluate(EncoderDecoder * encdec) {
  const auto evaluator = MTEval::EvaluatorFactory::create("BLEU");
  MTEval::Statistics stats;
  sampler_.rewind();
  
  while (sampler_.hasSamples()) {
    vector<nmtkit::Sample> samples = sampler_.getSamples();
    nmtkit::InferenceGraph ig = encdec->infer(
        samples[0].source,
        src_bos_id_, src_eos_id_,
        trg_bos_id_, trg_eos_id_,
        ::MAX_SEQUENCE_LENGTH, ::BEAM_WIDTH, ::WORD_PENALTY);
    const auto hyp_nodes = ig.findOneBestPath(trg_bos_id_, trg_eos_id_);

    // Make a resulting sequence without <s> and </s>.
    vector<unsigned> hyp_ids;
    for (unsigned i = 1; i < hyp_nodes.size() - 1; ++i) {
      hyp_ids.emplace_back(hyp_nodes[i]->label().word_id);
    }
    MTEval::Sample eval_sample {hyp_ids, {samples[0].target}};
    stats += evaluator->map(eval_sample);
  }

  return evaluator->integrate(stats);
}

} // namespace nmtkit

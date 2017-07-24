#include <nmtkit/loss_evaluator.h>

#include <vector>
#include <nmtkit/exception.h>

using std::string;
using std::vector;

namespace {
  const unsigned MAX_SEQUENCE_LENGTH = 1024;
  const float MAX_SEQUENCE_RATIO = 1024.0f;
  const unsigned BATCH_SIZE = 1;
} // namespace

namespace nmtkit {

LossEvaluator::LossEvaluator(
    const string & src_filepath,
    const string & trg_filepath,
    const Vocabulary & src_vocab,
    const Vocabulary & trg_vocab)
: converter_(src_vocab, trg_vocab)
, sampler_(
    src_filepath, trg_filepath,
    src_vocab, trg_vocab,
    ::MAX_SEQUENCE_LENGTH, ::MAX_SEQUENCE_RATIO, ::BATCH_SIZE) {}

float LossEvaluator::evaluate(EncoderDecoder * encdec) {
  unsigned num_outputs = 0;
  float total_loss = 0.0f;
  sampler_.rewind();

  while (sampler_.hasSamples()) {
    const vector<nmtkit::Sample> samples = sampler_.getSamples();
    const nmtkit::Batch batch = converter_.convert(samples);
    dynet::ComputationGraph cg;
    dynet::expr::Expression total_loss_expr = encdec->buildTrainGraph(
        batch, &cg, false);

    // batch includes words, <s> and </s>, and we only count words and </s>.
    num_outputs += batch.target_ids.size() - 1;
    total_loss += dynet::as_scalar(cg.forward(total_loss_expr));
  }

  NMTKIT_CHECK(num_outputs > 0, "Empty corpus is given.");
  return total_loss / num_outputs;
}

} // namespace nmtkit

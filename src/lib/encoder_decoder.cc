#include "config.h"

#include <nmtkit/encoder_decoder.h>

#include <algorithm>
#include <nmtkit/array.h>
#include <nmtkit/exception.h>

/* Input/output mapping for training/force decoding:
 *
 *  Encoder Inputs                      Decoder Inputs
 * +-----------------------------+     +-----------------------------+
 *  s[0], s[1], s[2], ..., s[n-1]  |||  t[0], t[1], t[2], ..., t[m-2], t[m-1]
 *                                           +-------------------------------+
 *                                            Decoder Outputs
 */

using namespace std;
using dynet::expr::Expression;

namespace DE = dynet::expr;

namespace nmtkit {

EncoderDecoder::EncoderDecoder(
    boost::shared_ptr<Encoder> & encoder,
    boost::shared_ptr<Decoder> & decoder,
    boost::shared_ptr<Attention> & attention,
    boost::shared_ptr<Predictor> & predictor)
: encoder_(encoder)
, decoder_(decoder)
, attention_(attention)
, predictor_(predictor) {
}

Expression EncoderDecoder::buildTrainGraph(
    const Batch & batch,
    const float dropout_ratio,
    dynet::ComputationGraph * cg) {
  // Encode
  encoder_->prepare(dropout_ratio, cg);
  const vector<Expression> enc_outputs = encoder_->compute(
      batch.source_ids, cg);

  // Decode
  attention_->prepare(enc_outputs, cg);
  predictor_->prepare(cg);
  Decoder::State state = decoder_->prepare(
      encoder_->getStates(), dropout_ratio, cg);
  vector<Expression> losses;

  for (unsigned i = 0; i < batch.target_ids.size() - 1; ++i) {
    Expression out_embed;
    state = decoder_->oneStep(
        state, batch.target_ids[i], attention_.get(), cg, nullptr, &out_embed);
    losses.emplace_back(
        predictor_->computeLoss(out_embed, batch.target_ids[i + 1]));
  }

  return DE::sum_batches(DE::sum(losses));
}

InferenceGraph EncoderDecoder::beamSearch(
    const unsigned bos_id,
    const unsigned eos_id,
    const unsigned max_length,
    const unsigned beam_width,
    const float word_penalty,
    dynet::ComputationGraph * cg) {
  // Candidates of new nodes.
  struct Candidate {
    InferenceGraph::Node * prev;
    InferenceGraph::Label label;
    Decoder::State state;
  };

  // Initialize the inference graph.
  InferenceGraph ig;
  vector<Candidate> history {
    // The "<s>" node
    {nullptr,
     {bos_id, 0.0f, 0.0f, {}},
     decoder_->prepare(encoder_->getStates(), 0.0f, cg)},
  };
  float best_accum_log_prob = -1e10f;

  for (unsigned length = 1; ; ++length) {
    vector<Candidate> next_cands;

    for (Candidate & prev : history) {
      // Add a new node using a history.
      auto prev_node = ig.addNode(prev.label);
      if (prev.prev != nullptr) {
        ig.connect(prev.prev, prev_node);
      }

      if (prev.label.word_id == eos_id) {
        // Reached a "</s>" node.
        // Try to replace the current best result.
        if (prev.label.accum_log_prob > best_accum_log_prob) {
          best_accum_log_prob = prev.label.accum_log_prob;
        }
      } else {
        // Expands the node.
        const vector<unsigned> inputs {prev.label.word_id};
        Expression atten_probs;
        Expression out_embed;
        Decoder::State next_state = decoder_->oneStep(
            prev.state, inputs, attention_.get(), cg, &atten_probs, &out_embed);

        // Obtains attention probabilities.
        const vector<dynet::real> atten_probs_values = dynet::as_vector(
            cg->incremental_forward(atten_probs));
        vector<float> out_atten_probs;
        for (const dynet::real p : atten_probs_values) {
          out_atten_probs.emplace_back(static_cast<float>(p));
        }

        // Predict next words.
        const vector<Predictor::Result> kbest =
            length < max_length ?
            predictor_->predictKBest(out_embed, beam_width, cg) :  // k-best
            predictor_->predictByIDs(out_embed, {eos_id}, cg);  // "</s>"

        // Make next candidates.
        for (const Predictor::Result & res : kbest) {
          next_cands.emplace_back(Candidate {
              prev_node,
              {res.word_id,
               res.log_prob + word_penalty,
               prev.label.accum_log_prob + res.log_prob + word_penalty,
               out_atten_probs},
              next_state,
          });
        }
      }
    }  // for (History & prev : history)

    // If there is no next candidates, all of previous nodes are "</s>".
    if (next_cands.empty()) {
      break;
    }

    // Obtains top-k candidates.
    vector<unsigned> kbest_ids = Array::kbest(
        next_cands,
        min(beam_width, static_cast<unsigned>(next_cands.size())),
        [](const Candidate & a, const Candidate & b) {
            return a.label.accum_log_prob > b.label.accum_log_prob;
        });

    // Finish the decoding if the probability of the top candidate is lower than
    // the best path.
    if (next_cands[kbest_ids[0]].label.accum_log_prob < best_accum_log_prob) {
      break;
    }

    // Make new previous nodes.
    history.clear();
    for (const unsigned id : kbest_ids){
      history.emplace_back(next_cands[id]);
    }
  }  // for (unsigned length = 1; ; ++length)

  return ig;
}

InferenceGraph EncoderDecoder::infer(
    const vector<unsigned> & source_ids,
    const unsigned bos_id,
    const unsigned eos_id,
    const unsigned max_length,
    const unsigned beam_width,
    const float word_penalty) {

  // Make batch data.
  vector<vector<unsigned>> source_ids_inner;
  source_ids_inner.emplace_back(vector<unsigned> {bos_id});
  for (const unsigned s : source_ids) {
    source_ids_inner.emplace_back(vector<unsigned> {s});
  }
  source_ids_inner.emplace_back(vector<unsigned> {eos_id});

  dynet::ComputationGraph cg;

  // Encode
  encoder_->prepare(0.0f, &cg);
  const vector<Expression> enc_outputs = encoder_->compute(
      source_ids_inner, &cg);

  // Infer output words
  attention_->prepare(enc_outputs, &cg);
  predictor_->prepare(&cg);
  return beamSearch(
      bos_id, eos_id, max_length, beam_width, word_penalty, &cg);
}

}  // namespace nmtkit

NMTKIT_SERIALIZATION_IMPL(nmtkit::EncoderDecoder);

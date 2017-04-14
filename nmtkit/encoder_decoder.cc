#include <config.h>

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

using dynet::expr::Expression;
using std::vector;

namespace DE = dynet::expr;

namespace nmtkit {

EncoderDecoder::EncoderDecoder(
    boost::shared_ptr<Encoder> & encoder,
    boost::shared_ptr<Decoder> & decoder,
    boost::shared_ptr<Attention> & attention,
    boost::shared_ptr<Predictor> & predictor,
    const std::string & loss_integration_type)
: encoder_(encoder)
, decoder_(decoder)
, attention_(attention)
, predictor_(predictor) {
  if (loss_integration_type == "sum") {
    mean_by_samples_ = false;
  } else if (loss_integration_type == "mean") {
    mean_by_samples_ = true;
  } else {
    NMTKIT_FATAL("Invalid loss_integration_type: " + loss_integration_type);
  }
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
        predictor_->computeLoss(out_embed, batch.target_ids[i + 1], cg));
  }

  // Calculates integrated loss value.
  float loss_divisor = 1.0f;
  if (mean_by_samples_) {
    loss_divisor *= batch.source_ids[0].size();
  }
  return DE::sum_batches(DE::sum(losses)) / loss_divisor;
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
        const vector<float> out_atten_probs = dynet::as_vector(
            cg->incremental_forward(atten_probs));

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
        std::min(beam_width, static_cast<unsigned>(next_cands.size())),
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
    const unsigned source_bos_id,
    const unsigned source_eos_id,
    const unsigned target_bos_id,
    const unsigned target_eos_id,
    const unsigned max_length,
    const unsigned beam_width,
    const float word_penalty) {

  // Make batch data.
  vector<vector<unsigned>> source_ids_inner {{source_bos_id}};
  for (const unsigned s : source_ids) {
    source_ids_inner.emplace_back(vector<unsigned> {s});
  }
  source_ids_inner.emplace_back(vector<unsigned> {source_eos_id});

  dynet::ComputationGraph cg;

  // Encode
  encoder_->prepare(0.0f, &cg);
  const vector<Expression> enc_outputs = encoder_->compute(
      source_ids_inner, &cg);

  // Infer output words
  attention_->prepare(enc_outputs, &cg);
  predictor_->prepare(&cg);
  return beamSearch(
      target_bos_id, target_eos_id, max_length, beam_width, word_penalty, &cg);
}

InferenceGraph EncoderDecoder::forceDecode(
    const vector<unsigned> & source_ids,
    const vector<unsigned> & target_ids,
    const unsigned source_bos_id,
    const unsigned source_eos_id,
    const unsigned target_bos_id,
    const unsigned target_eos_id) {

  // Make batch data.
  vector<vector<unsigned>> source_ids_inner {{source_bos_id}};
  vector<vector<unsigned>> target_ids_inner {{target_bos_id}};
  for (const unsigned s : source_ids) {
    source_ids_inner.emplace_back(vector<unsigned> {s});
  }
  for (const unsigned s : target_ids) {
    target_ids_inner.emplace_back(vector<unsigned> {s});
  }
  source_ids_inner.emplace_back(vector<unsigned> {source_eos_id});
  target_ids_inner.emplace_back(vector<unsigned> {target_eos_id});

  dynet::ComputationGraph cg;

  // Encode
  encoder_->prepare(0.0f, &cg);
  const vector<Expression> enc_outputs = encoder_->compute(
      source_ids_inner, &cg);

  // Prepare decoding.
  attention_->prepare(enc_outputs, &cg);
  predictor_->prepare(&cg);
  Decoder::State state = decoder_->prepare(encoder_->getStates(), 0.0f, &cg);
  InferenceGraph ig;
  InferenceGraph::Label prev_label {target_bos_id, 0.0, 0.0, {}};
  auto prev_node = ig.addNode(prev_label);

  // Do force decoding.
  for (unsigned i = 0; i < target_ids_inner.size() - 1; ++i) {
    // Expands the node.
    Expression atten_probs;
    Expression out_embed;
    state = decoder_->oneStep(
        state, target_ids_inner[i], attention_.get(),
        &cg, &atten_probs, &out_embed);

    // Obtains attention probabilities.
    const vector<float> out_atten_probs = dynet::as_vector(
        cg.incremental_forward(atten_probs));

    // Obtains next word.
    const vector<Predictor::Result> kbest = predictor_->predictByIDs(
        out_embed, {target_ids_inner[i+1][0]}, &cg);

    // Make next node.
    InferenceGraph::Label next_label {
      target_ids_inner[i+1][0],
      kbest[0].log_prob,
      prev_label.accum_log_prob + kbest[0].log_prob,
      out_atten_probs,
    };
    auto next_node = ig.addNode(next_label);
    ig.connect(prev_node, next_node);
    prev_label = std::move(next_label);
    prev_node = std::move(next_node);
  }

  return ig;
}

}  // namespace nmtkit

NMTKIT_SERIALIZATION_IMPL(nmtkit::EncoderDecoder);

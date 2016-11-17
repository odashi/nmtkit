#include "config.h"

#include <nmtkit/encoder_decoder.h>

#include <algorithm>
#include <nmtkit/array.h>
#include <nmtkit/backward_encoder.h>
#include <nmtkit/bidirectional_encoder.h>
#include <nmtkit/bilinear_attention.h>
#include <nmtkit/default_decoder.h>
#include <nmtkit/exception.h>
#include <nmtkit/forward_encoder.h>
#include <nmtkit/mlp_attention.h>
#include <nmtkit/softmax_predictor.h>

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
    unsigned src_vocab_size,
    unsigned trg_vocab_size,
    const string & encoder_type,
    unsigned src_embed_size,
    unsigned trg_embed_size,
    unsigned enc_hidden_size,
    unsigned dec_hidden_size,
    const string & atten_type,
    unsigned atten_size,
    dynet::Model * model) {
  NMTKIT_CHECK(src_vocab_size > 0, "src_vocab_size should be greater than 0.");
  NMTKIT_CHECK(trg_vocab_size > 0, "trg_vocab_size should be greater than 0.");
  NMTKIT_CHECK(src_embed_size > 0, "src_embed_size should be greater than 0.");
  NMTKIT_CHECK(trg_embed_size > 0, "trg_embed_size should be greater than 0.");
  NMTKIT_CHECK(
      enc_hidden_size > 0, "enc_hidden_size should be greater than 0.");
  NMTKIT_CHECK(
      dec_hidden_size > 0, "dec_hidden_size should be greater than 0.");
  // NOTE: atten_size would be checked in the attention selection section.

  // Encoder selection.
  if (encoder_type == "bidirectional") {
    encoder_.reset(
        new BidirectionalEncoder(
            1, src_vocab_size, src_embed_size, enc_hidden_size, model));
  } else if (encoder_type == "forward") {
    encoder_.reset(
        new ForwardEncoder(
            1, src_vocab_size, src_embed_size, enc_hidden_size, model));
  } else if (encoder_type == "backward") {
    encoder_.reset(
        new BackwardEncoder(
            1, src_vocab_size, src_embed_size, enc_hidden_size, model));
  } else {
    NMTKIT_FATAL("Invalid encoder type: " + encoder_type);
  }

  const unsigned context_size = encoder_->getStateSize();
  const unsigned enc_out_size = encoder_->getFinalStateSize();

  // Attention selection.
  if (atten_type == "mlp") {
    NMTKIT_CHECK(atten_size > 0, "atten_size should be greater than 0.");
    attention_.reset(
        new MLPAttention(context_size, dec_hidden_size, atten_size, model));
  } else if (atten_type == "bilinear") {
    attention_.reset(
        new BilinearAttention(context_size, dec_hidden_size, model));
  } else {
    NMTKIT_FATAL("Invalid attention type: " + atten_type);
  }

  // Decoder selection.
  decoder_.reset(
      new DefaultDecoder(
          trg_vocab_size, trg_embed_size, dec_hidden_size,
          enc_out_size, context_size, model));

  const unsigned dec_out_size = decoder_->getOutputSize();

  // Output projection.
  dec2logit_.reset(
      new MultilayerPerceptron({dec_out_size, trg_vocab_size}, model));

  predictor_.reset(new SoftmaxPredictor(trg_vocab_size));
}

vector<Expression> EncoderDecoder::buildDecoderGraph(
    const Expression & seed,
    const vector<vector<unsigned>> & target_ids,
    dynet::ComputationGraph * cg) {
  const unsigned tl = target_ids.size() - 1;
  vector<Expression> logits;
  Decoder::State state = decoder_->prepare(seed, cg);

  for (unsigned i = 0; i < tl; ++i) {
    Expression out_embed;
    state = decoder_->oneStep(
        state, target_ids[i], attention_.get(), cg, nullptr, &out_embed);
    logits.emplace_back(dec2logit_->compute(out_embed, cg));
  }

  return logits;
}

void EncoderDecoder::decodeForInference(
    const Expression & seed,
    const unsigned bos_id,
    const unsigned eos_id,
    const unsigned max_length,
    const unsigned beam_width,
    dynet::ComputationGraph * cg,
    InferenceGraph * ig) {
  // Candidates of new nodes.
  struct Candidate {
    InferenceGraph::Node * prev;
    InferenceGraph::Label label;
    Decoder::State state;
  };

  // Initialize the inference graph.
  ig->clear();
  vector<Candidate> history {
    // The "<s>" node
    {nullptr, {bos_id, 0.0f, 0.0f, {}}, decoder_->prepare(seed, cg)},
  };
  float best_accum_log_prob = -1e10f;

  for (unsigned length = 1; ; ++length) {
    vector<Candidate> next_cands;

    for (Candidate & prev : history) {
      // Add a new node using a history.
      auto prev_node = ig->addNode(prev.label);
      if (prev.prev != nullptr) {
        ig->connect(prev.prev, prev_node);
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
        Expression output;
        Decoder::State next_state = decoder_->oneStep(
            prev.state, inputs, attention_.get(), cg, &atten_probs, &output);

        // Obtains attention probabilities.
        const vector<dynet::real> atten_probs_values = dynet::as_vector(
            cg->incremental_forward(atten_probs));
        vector<float> out_atten_probs;
        for (const dynet::real p : atten_probs_values) {
          out_atten_probs.emplace_back(static_cast<float>(p));
        }

        // Predict next words.
        const Expression logit = dec2logit_->compute(output, cg);
        const vector<Predictor::Result> kbest =
            length < max_length ?
            predictor_->predictKBest(logit, beam_width, cg) :  // k-best words
            predictor_->predictByIDs(logit, {eos_id}, cg);  // only "</s>"

        // Make next candidates.
        for (const Predictor::Result & res : kbest) {
          next_cands.emplace_back(Candidate {
              prev_node,
              {res.word_id,
               res.log_prob,
               prev.label.accum_log_prob + res.log_prob,
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
}

Expression EncoderDecoder::buildTrainGraph(
    const Batch & batch,
    dynet::ComputationGraph * cg) {
  // Encode
  vector<Expression> enc_states;
  Expression enc_final_state;
  encoder_->build(batch.source_ids, cg, &enc_states, &enc_final_state);

  // Decode
  attention_->prepare(enc_states, cg);
  dec2logit_->prepare(cg);
  vector<Expression> logits = buildDecoderGraph(
      enc_final_state, batch.target_ids, cg);

  return predictor_->computeLoss(batch.target_ids, logits);
}

void EncoderDecoder::infer(
    const vector<unsigned> & source_ids,
    const unsigned bos_id,
    const unsigned eos_id,
    const unsigned max_length,
    const unsigned beam_width,
    dynet::ComputationGraph * cg,
    InferenceGraph * ig) {

  // Make batch data.
  vector<vector<unsigned>> source_ids_inner;
  source_ids_inner.emplace_back(vector<unsigned> {bos_id});
  for (const unsigned s : source_ids) {
    source_ids_inner.emplace_back(vector<unsigned> {s});
  }
  source_ids_inner.emplace_back(vector<unsigned> {eos_id});

  // Encode
  vector<Expression> enc_states;
  Expression enc_final_state;
  encoder_->build(source_ids_inner, cg, &enc_states, &enc_final_state);

  // Infer output words
  attention_->prepare(enc_states, cg);
  dec2logit_->prepare(cg);
  decodeForInference(
      enc_final_state, bos_id, eos_id, max_length, beam_width, cg, ig);
}

}  // namespace nmtkit

NMTKIT_SERIALIZATION_IMPL(nmtkit::EncoderDecoder);

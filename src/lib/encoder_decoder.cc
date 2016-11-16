#include "config.h"

#include <nmtkit/encoder_decoder.h>

#include <algorithm>
#include <nmtkit/array.h>
#include <nmtkit/backward_encoder.h>
#include <nmtkit/bidirectional_encoder.h>
#include <nmtkit/bilinear_attention.h>
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

  const unsigned mem_size = encoder_->getStateSize();
  const unsigned enc_out_size = encoder_->getFinalStateSize();
  const unsigned dec_in_size = trg_embed_size + mem_size;
  const unsigned dec_out_size = dec_hidden_size;
  // Note: In this implementation, encoder and decoder are connected through one
  //       nonlinear intermediate embedding layer. The size of this layer is
  //       determined using the average of both modules.
  const unsigned ie_size = (enc_out_size + dec_out_size) / 2;

  enc2dec_.reset(
      new MultilayerPerceptron({enc_out_size, ie_size, dec_out_size}, model));
  dec2logit_.reset(
      new MultilayerPerceptron({dec_out_size, trg_vocab_size}, model));

  // Attention selection.
  if (atten_type == "mlp") {
    NMTKIT_CHECK(atten_size > 0, "atten_size should be greater than 0.");
    attention_.reset(
        new MLPAttention(mem_size, dec_out_size, atten_size, model));
  } else if (atten_type == "bilinear") {
    attention_.reset(new BilinearAttention(mem_size, dec_out_size, model));
  } else {
    NMTKIT_FATAL("Invalid attention type: " + atten_type);
  }

  rnn_dec_.reset(new dynet::LSTMBuilder(1, dec_in_size, dec_out_size, model));

  predictor_.reset(new SoftmaxPredictor(trg_vocab_size));

  p_dec_lookup_ = model->add_lookup_parameters(
      trg_vocab_size, {trg_embed_size});
};

Expression EncoderDecoder::buildDecoderInitializerGraph(
    const Expression & enc_final_state,
    dynet::ComputationGraph * cg) {
  // NOTE: LSTMBuilder::start_new_sequence() takes initial states with below
  //       layout:
  //         {c1, c2, ..., cn, h1, h2, ..., hn}
  //       where cx is the initial cell states and hx is the initial outputs.
  Expression dec_init_c = enc2dec_->compute(
      enc2dec_->prepare(cg), enc_final_state, cg);
  Expression dec_init_h = DE::tanh(dec_init_c);
  rnn_dec_->new_graph(*cg);
  rnn_dec_->start_new_sequence({dec_init_c, dec_init_h});
  return dec_init_h;
}

vector<Expression> EncoderDecoder::buildDecoderGraph(
    const Expression & dec_init_h,
    const vector<vector<unsigned>> & target_ids,
    dynet::ComputationGraph * cg) {
  const unsigned tl = target_ids.size() - 1;
  Expression dec_h = dec_init_h;
  vector<Expression> dec2logit_params = dec2logit_->prepare(cg);
  vector<Expression> logits;

  for (unsigned i = 0; i < tl; ++i) {
    // Embedding
    const Expression embed = DE::lookup(*cg, p_dec_lookup_, target_ids[i]);

    // Attention
    const Expression context = attention_->compute(dec_h, cg)[1];

    // Decode
    dec_h = rnn_dec_->add_input(DE::concatenate({embed, context}));
    Expression logit = dec2logit_->compute(dec2logit_params, dec_h, cg);
    logits.emplace_back(logit);
  }

  return logits;
}

void EncoderDecoder::decodeForInference(
    const Expression & dec_init_h,
    const unsigned bos_id,
    const unsigned eos_id,
    const unsigned max_length,
    const unsigned beam_width,
    dynet::ComputationGraph * cg,
    InferenceGraph * ig) {
  const vector<Expression> dec2logit_params = dec2logit_->prepare(cg);

  // Candidates of new nodes.
  struct Candidate {
    InferenceGraph::Node * prev;
    InferenceGraph::Label label;
    Expression dec_h;
  };

  // Initialize the inference graph.
  ig->clear();
  vector<Candidate> history {
    // The "<s>" node
    {nullptr, {bos_id, 0.0f, 0.0f, {}}, dec_init_h},
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

        // Embedding
        const vector<unsigned> inputs {prev.label.word_id};
        const Expression embed = DE::lookup(*cg, p_dec_lookup_, inputs);

        // Attention
        const vector<Expression> atten_info = attention_->compute(
            prev.dec_h, cg);
        const vector<dynet::real> atten_probs_values = dynet::as_vector(
            cg->incremental_forward(atten_info[0]));
        vector<float> out_atten_probs;
        for (const dynet::real p : atten_probs_values) {
          out_atten_probs.emplace_back(static_cast<float>(p));
        }

        // Decode
        const Expression dec_h = rnn_dec_->add_input(
            DE::concatenate({embed, atten_info[1]}));
        const Expression logit = dec2logit_->compute(
            dec2logit_params, dec_h, cg);

        // Predict next words.
        const vector<Predictor::Result> kbest =
            length < max_length ?
            predictor_->predictKBest(logit, beam_width, cg) :  // k-best words
            predictor_->predictByIDs(logit, {eos_id}, cg);  // only "</s>"

        // Register next nodes.
        for (const Predictor::Result & res : kbest) {
          next_cands.emplace_back(Candidate {
              prev_node,
              {res.word_id,
               res.log_prob,
               prev.label.accum_log_prob + res.log_prob,
               out_atten_probs},
              dec_h,
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

  // Initialize attention
  attention_->prepare(enc_states, cg);

  // Decode
  Expression dec_init_h = buildDecoderInitializerGraph(enc_final_state, cg);
  vector<Expression> logits = buildDecoderGraph(
      dec_init_h, batch.target_ids, cg);

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

  // Initialize attention
  attention_->prepare(enc_states, cg);

  // Infer output words
  Expression dec_init_h = buildDecoderInitializerGraph(enc_final_state, cg);
  decodeForInference(
      dec_init_h, bos_id, eos_id, max_length, beam_width, cg, ig);
}

}  // namespace nmtkit

NMTKIT_SERIALIZATION_IMPL(nmtkit::EncoderDecoder);

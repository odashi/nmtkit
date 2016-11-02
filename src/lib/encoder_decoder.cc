#include "config.h"

#include <nmtkit/encoder_decoder.h>

#include <nmtkit/bidirectional_encoder.h>
#include <nmtkit/bilinear_attention.h>
#include <nmtkit/exception.h>
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

  encoder_.reset(
      new BidirectionalEncoder(
          1, src_vocab_size, src_embed_size, enc_hidden_size, model));
  
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
    const vector<Expression> & atten_info,
    const vector<vector<unsigned>> & target_ids,
    dynet::ComputationGraph * cg) {
  const unsigned tl = target_ids.size() - 1;
  Expression dec_h = dec_init_h;
  vector<Expression> dec2logit_params = dec2logit_->prepare(cg);
  vector<Expression> logits;

  for (unsigned i = 0; i < tl; ++i) {
    // Embedding
    Expression embed = DE::lookup(*cg, p_dec_lookup_, target_ids[i]);

    // Attention
    Expression context;
    attention_->compute(atten_info, dec_h, cg, nullptr, &context);

    // Decode
    dec_h = rnn_dec_->add_input(DE::concatenate({embed, context}));
    Expression logit = dec2logit_->compute(dec2logit_params, dec_h, cg);
    logits.emplace_back(logit);
  }

  return logits;
}

void EncoderDecoder::decodeForInference(
    const Expression & dec_init_h,
    const vector<Expression> & atten_info,
    const unsigned bos_id,
    const unsigned eos_id,
    const unsigned max_length,
    dynet::ComputationGraph * cg,
    InferenceGraph * ig) {
  ig->clear();
  InferenceGraph::Node * prev_node = ig->addNode({bos_id, 0.0f, {}});
  Expression dec_h = dec_init_h;
  vector<Expression> dec2logit_params = dec2logit_->prepare(cg);

  for (unsigned generated = 0; ; ++generated) {
    // Embedding
    vector<unsigned> inputs {prev_node->label().word_id};
    Expression embed = DE::lookup(*cg, p_dec_lookup_, inputs);

    // Attention
    Expression atten_probs, context;
    attention_->compute(atten_info, dec_h, cg, &atten_probs, &context);
    vector<dynet::real> atten_probs_values = dynet::as_vector(
        cg->incremental_forward(atten_probs));

    // Decode
    dec_h = rnn_dec_->add_input(DE::concatenate({embed, context}));
    Expression logit = dec2logit_->compute(dec2logit_params, dec_h, cg);

    // Predict next words.
    vector<Predictor::Result> next_words;
    if (generated < max_length - 1) {
      next_words = predictor_->predictKBest(logit, 1, cg);
    } else {
      next_words = predictor_->predictByIDs(logit, {eos_id}, cg);
    }

    // Make attention matrix.
    vector<float> out_atten_probs;
    for (const dynet::real p : atten_probs_values) {
      out_atten_probs.emplace_back(static_cast<float>(p));
    }

    // Make new graph nodes.
    InferenceGraph::Node * next_node = ig->addNode({
        next_words[0].word_id, next_words[0].log_prob, out_atten_probs});
    ig->connect(prev_node, next_node);

    // Go ahead or finish.
    prev_node = next_node;
    if (next_words[0].word_id == eos_id) {
      break;
    }
  }
}

Expression EncoderDecoder::buildTrainGraph(
    const Batch & batch,
    dynet::ComputationGraph * cg) {
  // Encode
  vector<Expression> enc_states;
  Expression enc_final_state;
  encoder_->build(batch.source_ids, cg, &enc_states, &enc_final_state);

  // Initialize attention
  vector<Expression> atten_info = attention_->prepare(enc_states, cg);

  // Decode
  Expression dec_init_h = buildDecoderInitializerGraph(enc_final_state, cg);
  vector<Expression> logits = buildDecoderGraph(
      dec_init_h, atten_info, batch.target_ids, cg);

  return predictor_->computeLoss(batch.target_ids, logits);
}

void EncoderDecoder::infer(
    const vector<unsigned> & source_ids,
    const unsigned bos_id,
    const unsigned eos_id,
    const unsigned max_length,
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
  vector<Expression> atten_info = attention_->prepare(enc_states, cg);

  // Infer output words
  Expression dec_init_h = buildDecoderInitializerGraph(enc_final_state, cg);
  decodeForInference(
      dec_init_h, atten_info, bos_id, eos_id, max_length, cg, ig);
}

}  // namespace nmtkit

NMTKIT_SERIALIZATION_IMPL(nmtkit::EncoderDecoder);

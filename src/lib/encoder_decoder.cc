#include <nmtkit/encoder_decoder.h>

/* Input/output mapping for training/force decoding:
 *
 *  Encoder Inputs                      Decoder Inputs
 * +-----------------------------+     +-----------------------------+
 *  s[0], s[1], s[2], ..., s[n-1]  |||  t[0], t[1], t[2], ..., t[m-2], t[m-1]
 *                                           +-------------------------------+
 *                                            Decoder Outputs
 */

#include <nmtkit/array.h>

using namespace std;

namespace DE = dynet::expr;

namespace nmtkit {

EncoderDecoder::EncoderDecoder(
    unsigned src_vocab_size,
    unsigned trg_vocab_size,
    unsigned embed_size,
    unsigned hidden_size,
    dynet::Model * model)
: rnn_fw_enc_(1, embed_size, hidden_size, model),
  rnn_bw_enc_(1, embed_size, hidden_size, model),
  rnn_dec_(1, embed_size, hidden_size, model) {
  // Note: Intermediate embedding between encoder/decoder has 1.5 times larger
  //       layer:
  //         Encoder ............. hidden_size (fw) + hidden_size (bw)
  //           -> Intermediate ... 1.5 * hidden_size
  //                -> Decoder ... hidden_size
  const unsigned ie_size = static_cast<unsigned>(hidden_size * 1.5);

  p_enc_lookup_ = model->add_lookup_parameters(src_vocab_size, {embed_size});
  p_dec_lookup_ = model->add_lookup_parameters(trg_vocab_size, {embed_size});
  p_enc2ie_w_ = model->add_parameters({ie_size, 2 * hidden_size});
  p_enc2ie_b_ = model->add_parameters({ie_size});
  p_ie2dec_w_ = model->add_parameters({hidden_size, ie_size});
  p_ie2dec_b_ = model->add_parameters({hidden_size});
  p_dec2out_w_ = model->add_parameters({trg_vocab_size, hidden_size});
  p_dec2out_b_ = model->add_parameters({trg_vocab_size});
};

void EncoderDecoder::buildEncoderGraph(
    const vector<vector<unsigned>> & source_ids,
    dynet::ComputationGraph * cg,
    vector<DE::Expression> * fw_enc_outputs,
    vector<DE::Expression> * bw_enc_outputs) {
  fw_enc_outputs->clear();
  bw_enc_outputs->clear();
  const int sl = source_ids.size();

  // Embedding lookup
  vector<DE::Expression> embeds;
  for (int i = 0; i < sl; ++i) {
    embeds.emplace_back(DE::lookup(*cg, p_enc_lookup_, source_ids[i]));
  }

  // Forward encoding
  rnn_fw_enc_.new_graph(*cg);
  rnn_fw_enc_.start_new_sequence();
  for (int i = 0; i < sl; ++i) {
    fw_enc_outputs->emplace_back(rnn_fw_enc_.add_input(embeds[i]));
  }

  // Backward encoding
  rnn_bw_enc_.new_graph(*cg);
  rnn_bw_enc_.start_new_sequence();
  for (int i = sl - 1; i >= 0; --i) {
    bw_enc_outputs->emplace_back(rnn_bw_enc_.add_input(embeds[i]));
  }
  Array::reverse(bw_enc_outputs);
}

void EncoderDecoder::buildDecoderInitializerGraph(
    const vector<DE::Expression> & fw_enc_outputs,
    const vector<DE::Expression> & bw_enc_outputs,
    dynet::ComputationGraph * cg,
    vector<DE::Expression> * dec_init_states) {
  dec_init_states->clear();

  // Encoder -> intermediate node
  DE::Expression enc2ie_w = DE::parameter(*cg, p_enc2ie_w_);
  DE::Expression enc2ie_b = DE::parameter(*cg, p_enc2ie_b_);
  DE::Expression enc_final_h = DE::concatenate(
      {fw_enc_outputs.back(), bw_enc_outputs.front()});
  DE::Expression ie_u = enc2ie_w * enc_final_h + enc2ie_b;
  DE::Expression ie_h = DE::rectify(ie_u);

  // Intermediate node -> decoder
  DE::Expression ie2dec_w = DE::parameter(*cg, p_ie2dec_w_);
  DE::Expression ie2dec_b = DE::parameter(*cg, p_ie2dec_b_);
  DE::Expression dec_init_c = ie2dec_w * ie_h + ie2dec_b;
  DE::Expression dec_init_h = DE::tanh(dec_init_c);

  // NOTE: LSTMBuilder::start_new_sequence() takes initial states with below
  //       layout:
  //         {c1, c2, ..., cn, h1, h2, ..., hn}
  //       where cx is the initial cell states and hx is the initial outputs.
  dec_init_states->emplace_back(dec_init_c);
  dec_init_states->emplace_back(dec_init_h);
}

void EncoderDecoder::buildDecoderGraph(
    const vector<DE::Expression> & dec_init_states,
    const vector<vector<unsigned>> & target_ids,
    dynet::ComputationGraph * cg,
    vector<DE::Expression> * dec_outputs) {
  dec_outputs->clear();
  const unsigned tl = target_ids.size() - 1;

  rnn_dec_.new_graph(*cg);
  rnn_dec_.start_new_sequence(dec_init_states);
  DE::Expression dec2out_w = DE::parameter(*cg, p_dec2out_w_);
  DE::Expression dec2out_b = DE::parameter(*cg, p_dec2out_b_);

  for (unsigned i = 0; i < tl; ++i) {
    DE::Expression embed = DE::lookup(*cg, p_dec_lookup_, target_ids[i]);
    DE::Expression dec_h = rnn_dec_.add_input(embed);
    DE::Expression dec_out = dec2out_w * dec_h + dec2out_b;
    dec_outputs->emplace_back(dec_out);
  }
}

void EncoderDecoder::decodeForInference(
    const vector<DE::Expression> & dec_init_states,
    const unsigned bos_id,
    const unsigned eos_id,
    const unsigned max_length,
    dynet::ComputationGraph * cg,
    vector<InferenceGraph::Node> * outputs) {
  outputs->clear();

  rnn_dec_.new_graph(*cg);
  rnn_dec_.start_new_sequence(dec_init_states);
  DE::Expression dec2out_w = DE::parameter(*cg, p_dec2out_w_);
  DE::Expression dec2out_b = DE::parameter(*cg, p_dec2out_b_);

  outputs->emplace_back(InferenceGraph::Node {bos_id, 0.0f});

  while (outputs->back().word_id != eos_id && outputs->size() <= max_length) {
    vector<unsigned> inputs {outputs->back().word_id};
    DE::Expression embed = DE::lookup(*cg, p_dec_lookup_, inputs);
    DE::Expression dec_h = rnn_dec_.add_input(embed);
    DE::Expression dec_out = dec2out_w * dec_h + dec2out_b;
    DE::Expression log_probs_expr = DE::log_softmax(dec_out);
    vector<dynet::real> log_probs = dynet::as_vector(
        cg->incremental_forward(log_probs_expr));

    unsigned output_id = eos_id;
    if (outputs->size() < max_length) {
      output_id = Array::argmax(log_probs);
    }
    outputs->emplace_back(InferenceGraph::Node {
        output_id,
        static_cast<float>(log_probs[output_id])});
  }
}

void EncoderDecoder::buildLossGraph(
    const vector<vector<unsigned>> & target_ids,
    const vector<DE::Expression> & dec_outputs,
    vector<DE::Expression> * losses) {
  losses->clear();
  const unsigned tl = target_ids.size() - 1;

  for (unsigned i = 0; i < tl; ++i) {
    DE::Expression loss = DE::pickneglogsoftmax(
        dec_outputs[i], target_ids[i + 1]);
    losses->emplace_back(loss);
  }
}

DE::Expression EncoderDecoder::buildTrainGraph(
    const Batch & batch,
    dynet::ComputationGraph * cg) {
  // Encode
  vector<DE::Expression> fw_enc_outputs, bw_enc_outputs;
  buildEncoderGraph(batch.source_ids, cg, &fw_enc_outputs, &bw_enc_outputs);

  // Initialize decoder
  vector<DE::Expression> dec_init_states;
  buildDecoderInitializerGraph(
      fw_enc_outputs, bw_enc_outputs, cg, &dec_init_states);

  // Decode
  vector<DE::Expression> dec_outputs;
  buildDecoderGraph(dec_init_states, batch.target_ids, cg, &dec_outputs);

  // Calculate losses
  vector<DE::Expression> losses;
  buildLossGraph(batch.target_ids, dec_outputs, &losses);
  DE::Expression total_loss = DE::sum_batches(DE::sum(losses));
  return total_loss;
}

void EncoderDecoder::infer(
    const vector<unsigned> & source_ids,
    const unsigned bos_id,
    const unsigned eos_id,
    const unsigned max_length,
    dynet::ComputationGraph * cg,
    vector<InferenceGraph::Node> * outputs) {

  // Make batch data.
  vector<vector<unsigned>> source_ids_inner;
  source_ids_inner.emplace_back(vector<unsigned> {bos_id});
  for (const unsigned s : source_ids) {
    source_ids_inner.emplace_back(vector<unsigned> {s});
  }
  source_ids_inner.emplace_back(vector<unsigned> {eos_id});

  // Encode
  vector<DE::Expression> fw_enc_outputs, bw_enc_outputs;
  buildEncoderGraph(source_ids_inner, cg, &fw_enc_outputs, &bw_enc_outputs);

  // Initialize decoder
  vector<DE::Expression> dec_init_states;
  buildDecoderInitializerGraph(
      fw_enc_outputs, bw_enc_outputs, cg, &dec_init_states);

  // Infer output words
  decodeForInference(dec_init_states, bos_id, eos_id, max_length, cg, outputs);
}

}  // namespace nmtkit


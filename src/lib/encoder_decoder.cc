#include <nmtkit/encoder_decoder.h>

#include <iostream>
#include <nmtkit/array.h>
#include <nmtkit/exception.h>

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
  NMTKIT_CHECK(fw_enc_outputs->empty(), "fw_enc_outputs is not empty.");
  NMTKIT_CHECK(bw_enc_outputs->empty(), "bw_enc_outputs is not empty.");
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
  NMTKIT_CHECK(dec_init_states->empty(), "dec_init_states is not empty.");

  DE::Expression enc2ie_w = DE::parameter(*cg, p_enc2ie_w_);
  DE::Expression enc2ie_b = DE::parameter(*cg, p_enc2ie_b_);
  DE::Expression enc_final_h = DE::concatenate(
      {fw_enc_outputs.back(), bw_enc_outputs.front()});
  DE::Expression ie_u = enc2ie_w * enc_final_h + enc2ie_b;
  DE::Expression ie_h = DE::rectify(ie_u);

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

DE::Expression EncoderDecoder::buildTrainGraph(
    const Batch & batch,
    dynet::ComputationGraph * cg) {
  const auto & trg = batch.target_ids;
  const int tl = trg.size();

  vector<DE::Expression> fw_enc_outputs, bw_enc_outputs;
  buildEncoderGraph(batch.source_ids, cg, &fw_enc_outputs, &bw_enc_outputs);

  vector<DE::Expression> dec_init_states;
  buildDecoderInitializerGraph(
      fw_enc_outputs, bw_enc_outputs, cg, &dec_init_states);

  // Decoding
  rnn_dec_.new_graph(*cg);
  rnn_dec_.start_new_sequence(dec_init_states);
  DE::Expression dec2out_w = DE::parameter(*cg, p_dec2out_w_);
  DE::Expression dec2out_b = DE::parameter(*cg, p_dec2out_b_);
  vector<DE::Expression> losses;

  // NOTE: First target words are used only the inputs, and final target words
  //       are used only the outputs.
  for (int i = 0; i < tl - 1; ++i) {
    DE::Expression dec_embed = DE::lookup(*cg, p_dec_lookup_, trg[i]);
    DE::Expression dec_h = rnn_dec_.add_input(dec_embed);
    DE::Expression dec_out = dec2out_w * dec_h + dec2out_b;
    DE::Expression loss = DE::pickneglogsoftmax(dec_out, trg[i + 1]);
    losses.push_back(loss);
  }

  DE::Expression total_loss = DE::sum_batches(DE::sum(losses));
  return total_loss;
}

}  // namespace nmtkit


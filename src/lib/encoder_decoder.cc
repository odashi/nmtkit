#include <nmtkit/encoder_decoder.h>

#include <iostream>

using namespace std;

namespace DE = dynet::expr;

namespace nmtkit {

EncoderDecoder::EncoderDecoder(
    unsigned src_vocab_size,
    unsigned trg_vocab_size,
    unsigned embed_size,
    unsigned hidden_size,
    dynet::Model * model)
: rnn_bw_enc_(1, embed_size, hidden_size, model),
  rnn_dec_(1, embed_size, hidden_size, model) {
  // Note: Intermediate embedding between encoder/decoder has 1.5 times larger
  //       layer.
  const unsigned ie_size = static_cast<unsigned>(hidden_size * 1.5);

  p_enc_lookup_ = model->add_lookup_parameters(src_vocab_size, {embed_size});
  p_dec_lookup_ = model->add_lookup_parameters(trg_vocab_size, {embed_size});
  p_enc2ie_w_ = model->add_parameters({ie_size, hidden_size});
  p_enc2ie_b_ = model->add_parameters({ie_size});
  p_ie2dec_w_ = model->add_parameters({hidden_size, ie_size});
  p_ie2dec_b_ = model->add_parameters({hidden_size});
  p_dec2out_w_ = model->add_parameters({trg_vocab_size, hidden_size});
  p_dec2out_b_ = model->add_parameters({trg_vocab_size});
};

Expression EncoderDecoder::buildTrainGraph(
    const Batch & batch,
    dynet::ComputationGraph * cg) {
  const auto & src = batch.source_id;
  const auto & trg = batch.target_id;
  const int sl = src.size();
  const int tl = trg.size();

  // Backward encoding
  rnn_bw_enc_.new_graph(*cg);
  rnn_bw_enc_.start_new_sequence();
  
  for (int i = sl - 1; i >= 0; --i) {
    Expression enc_embed = DE::lookup(*cg, p_enc_lookup_, src[i]);
    rnn_bw_enc_.add_input(enc_embed);
  }

  // Encoder to decoder transition
  Expression enc2ie_w = DE::parameter(*cg, p_enc2ie_w_);
  Expression enc2ie_b = DE::parameter(*cg, p_enc2ie_b_);
  Expression bw_enc_final_h = rnn_bw_enc_.final_h()[0];
  Expression ie_u = enc2ie_w * bw_enc_final_h + enc2ie_b;
  Expression ie_h = DE::rectify(ie_u);
  
  Expression ie2dec_w = DE::parameter(*cg, p_ie2dec_w_);
  Expression ie2dec_b = DE::parameter(*cg, p_ie2dec_b_);
  Expression dec_init_c = ie2dec_w * ie_h + ie2dec_b;
  Expression dec_init_h = DE::tanh(dec_init_c);

  // Decoding
  rnn_dec_.new_graph(*cg);
  // NOTE: start_new_sequence() takes initial states with below layout:
  //       {c1, c2, ..., cn, h1, h2, ..., hn}
  rnn_dec_.start_new_sequence({dec_init_c, dec_init_h});
  Expression dec2out_w = DE::parameter(*cg, p_dec2out_w_);
  Expression dec2out_b = DE::parameter(*cg, p_dec2out_b_);
  vector<Expression> losses;

  // NOTE: First target words are used only the inputs, and final target words
  //       are used only the outputs.
  for (int i = 0; i < tl - 1; ++i) {
    Expression dec_embed = DE::lookup(*cg, p_dec_lookup_, trg[i]);
    Expression dec_h = rnn_dec_.add_input(dec_embed);
    Expression dec_out = dec2out_w * dec_h + dec2out_b;
    Expression loss = DE::pickneglogsoftmax(dec_out, trg[i + 1]);
    losses.push_back(loss);
  }
  
  Expression total_loss = DE::sum_batches(DE::sum(losses));
  return total_loss;
}

}  // namespace nmtkit


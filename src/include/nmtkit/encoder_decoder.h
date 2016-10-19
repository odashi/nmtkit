#ifndef NMTKIT_ENCODER_DECODER_H_
#define NMTKIT_ENCODER_DECODER_H_

#include <string>
#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/lstm.h>
#include <dynet/model.h>
#include <nmtkit/basic_types.h>

namespace nmtkit {

class EncoderDecoder {
  EncoderDecoder() = delete;
  EncoderDecoder(const EncoderDecoder &) = delete;
  EncoderDecoder(EncoderDecoder &&) = delete;
  EncoderDecoder & operator=(const EncoderDecoder &) = delete;
  EncoderDecoder & operator=(EncoderDecoder &&) = delete;

public:
  // Constructs a new encoder-decoder model.
  // Arguments:
  //   src_vocab_size: Source vocabulary size.
  //   trg_vocab_size: Target vocabulary size.
  //   embed_size: Embedding layer size.
  //   hidden_size: RNN hidden layer size.
  //   atten_size: Attention MLP hidden layer size.
  EncoderDecoder(
      unsigned src_vocab_size,
      unsigned trg_vocab_size,
      unsigned embed_size,
      unsigned hidden_size,
      dynet::Model * model);

  // Constructs computation graph for the batch data.
  // Arguments:
  //   batch: Batch data to be trained.
  //   cg: Target computation graph.
  //
  // Returns:
  //   dynet::Expression object representing total loss value.
  dynet::expr::Expression buildTrainGraph(
      const Batch & batch,
      dynet::ComputationGraph * cg);

private:
  dynet::LSTMBuilder rnn_fw_enc_;
  dynet::LSTMBuilder rnn_bw_enc_;
  dynet::LSTMBuilder rnn_dec_;
  dynet::LookupParameter p_enc_lookup_;
  dynet::LookupParameter p_dec_lookup_;
  dynet::Parameter p_enc2ie_w_;
  dynet::Parameter p_enc2ie_b_;
  dynet::Parameter p_ie2dec_w_;
  dynet::Parameter p_ie2dec_b_;
  dynet::Parameter p_dec2out_w_;
  dynet::Parameter p_dec2out_b_;
};

}  // namespace nmtkit

#endif  // NMTKIT_ENCODER_DECODER_H_


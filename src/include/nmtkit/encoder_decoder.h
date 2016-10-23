#ifndef NMTKIT_ENCODER_DECODER_H_
#define NMTKIT_ENCODER_DECODER_H_

#include <string>
#include <vector>
#include <boost/serialization/serialization.hpp>
#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/lstm.h>
#include <dynet/model.h>
#include <nmtkit/basic_types.h>
#include <nmtkit/inference_graph.h>

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

  // Generates output sentence using given input sentence.
  // Arguments:
  //   source_ids: List of source word IDs.
  //   bos_id: "<s>" ID in the target language.
  //   eos_id: "</s>" ID in the target language.
  //   max_length: Maximum number of words (except "<s>") to be generated.
  //   cg: Target computation graph.
  //   ig: Placeholder of the output inference graph.
  void infer(
      const std::vector<unsigned> & source_ids,
      const unsigned bos_id,
      const unsigned eos_id,
      const unsigned max_length,
      dynet::ComputationGraph * cg,
      InferenceGraph * ig);

private:
  // Constructs encoder graph.
  // Arguments:
  //   source_ids: List of source word IDs.
  //   cg: Target computation graph.
  //   fw_enc_outputs: Placeholder of the forward encoder outputs.
  //   bw_enc_outputs: Placeholder of the backward encoder outputs.
  void buildEncoderGraph(
      const std::vector<std::vector<unsigned>> & source_ids,
      dynet::ComputationGraph * cg,
      std::vector<dynet::expr::Expression> * fw_enc_outputs,
      std::vector<dynet::expr::Expression> * bw_enc_outputs);

  // Constructs decoder initializer graph.
  // Arguments:
  //   fw_enc_outputs: Forward encoder states.
  //   bw_enc_outputs: Backward encoder states.
  //   cg: Target computation graph.
  //   dec_init_states: Placeholder of the initial decoder states.
  void buildDecoderInitializerGraph(
      const std::vector<dynet::expr::Expression> & fw_enc_outputs,
      const std::vector<dynet::expr::Expression> & bw_enc_outputs,
      dynet::ComputationGraph * cg,
      std::vector<dynet::expr::Expression> * dec_init_states);

  // Constructs decoder graph for training.
  // Arguments:
  //   dec_init_states: List of decoder initial states.
  //   target_ids: Target word IDs for this step.
  //   cg: Target computation graph.
  //   dec_outputs: Placeholder of the decoder output distributions.
  void buildDecoderGraph(
      const std::vector<dynet::expr::Expression> & dec_init_states,
      const std::vector<std::vector<unsigned>> & target_ids,
      dynet::ComputationGraph * cg,
      std::vector<dynet::expr::Expression> * dec_outputs);

  // Constructs graph of the output loss function.
  // Arguments:
  //   target_ids: Target word IDs for this step.
  //   dec_outputs: Decoder output distributions.
  //   losses: Placeholder of the loss expressions.
  void buildLossGraph(
      const std::vector<std::vector<unsigned>> & target_ids,
      const std::vector<dynet::expr::Expression> & dec_outputs,
      std::vector<dynet::expr::Expression> * losses);

  // Generates output sequence using encoder results.
  // Arguments:
  //   dec_init_states: List of decoder initial states.
  //   bos_id: "<s>" ID in the target language.
  //   eos_id: "</s>" ID in the target language.
  //   max_length: Maximum number of words (except "<s>") to be generated.
  //   cg: Target computation graph.
  //   ig: Placeholder of the output inference graph.
  void decodeForInference(
      const std::vector<dynet::expr::Expression> & dec_init_states,
      const unsigned bos_id,
      const unsigned eos_id,
      const unsigned max_length,
      dynet::ComputationGraph * cg,
      InferenceGraph * ig);

  // Boost serialization interface.
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive & ar, const unsigned int) {
    ar & rnn_fw_enc_ & rnn_bw_enc_ & rnn_dec_;
    ar & p_enc_lookup_ & p_dec_lookup_;
    ar & p_enc2ie_w_ & p_enc2ie_b_;
    ar & p_ie2dec_w_ & p_ie2dec_b_;
    ar & p_dec2out_w_ & p_dec2out_b_;
  }

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


#ifndef NMTKIT_ENCODER_DECODER_H_
#define NMTKIT_ENCODER_DECODER_H_

#include <string>
#include <vector>
#include <boost/scoped_ptr.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/scoped_ptr.hpp>
#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/lstm.h>
#include <dynet/model.h>
#include <nmtkit/attention.h>
#include <nmtkit/basic_types.h>
#include <nmtkit/encoder.h>
#include <nmtkit/inference_graph.h>
#include <nmtkit/multilayer_perceptron.h>
#include <nmtkit/serialization_utils.h>

namespace nmtkit {

class EncoderDecoder {
  EncoderDecoder(const EncoderDecoder &) = delete;
  EncoderDecoder(EncoderDecoder &&) = delete;
  EncoderDecoder & operator=(const EncoderDecoder &) = delete;
  EncoderDecoder & operator=(EncoderDecoder &&) = delete;

public:
  // Constructs an empty encoder-decoder model.
  EncoderDecoder() {}
  
  // Constructs a new encoder-decoder model.
  //
  // Arguments:
  //   src_vocab_size: Source vocabulary size.
  //   trg_vocab_size: Target vocabulary size.
  //   src_embed_size: Number of units in source embedding layer.
  //   trg_embed_size: Number of units in target embedding layer.
  //   enc_hidden_size: Number of units in encoder states.
  //   dec_hidden_size: Number of units in decoder states.
  //   atten_type: Name of the attention estimator.
  //               Available values:
  //                 "mlp": Multilayer perceptron-based attention
  //                        (similar to [Bahdanau+14])
  //                 "bilinear": Bilinear attention
  //                             ("general" method in [Luong+15])
  //   atten_size: Number of units in attention hidden layer.
  //               This parameter is used only in "mlp" attention.
  //   model: Model object for training.
  EncoderDecoder(
      unsigned src_vocab_size,
      unsigned trg_vocab_size,
      unsigned src_embed_size,
      unsigned trg_embed_size,
      unsigned enc_hidden_size,
      unsigned dec_hidden_size,
      const std::string & atten_type,
      unsigned atten_size,
      dynet::Model * model);

  // Constructs computation graph for the batch data.
  //
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
  //
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
  // Constructs decoder initializer graph.
  //
  // Arguments:
  //   enc_final_state: Final state of the encoder.
  //   cg: Target computation graph.
  //
  // Returns:
  //   Initial output value of the decoder.
  dynet::expr::Expression buildDecoderInitializerGraph(
      const dynet::expr::Expression & enc_final_state,
      dynet::ComputationGraph * cg);

  // Constructs decoder graph for training.
  //
  // Arguments:
  //   dec_init_h: Initial output value of the decoder.
  //   atten_info: Precomputed values for the attention estimator.
  //   target_ids: Target word IDs for this step.
  //   cg: Target computation graph.
  //   dec_outputs: Placeholder of the decoder output distributions.
  void buildDecoderGraph(
      const dynet::expr::Expression & dec_init_h,
      const std::vector<dynet::expr::Expression> & atten_info,
      const std::vector<std::vector<unsigned>> & target_ids,
      dynet::ComputationGraph * cg,
      std::vector<dynet::expr::Expression> * dec_outputs);

  // Generates output sequence using encoder results.
  //
  // Arguments:
  //   dec_init_h: Initial output value of the decoder.
  //   atten_info: Precomputed values for the attention estimator.
  //   bos_id: "<s>" ID in the target language.
  //   eos_id: "</s>" ID in the target language.
  //   max_length: Maximum number of words (except "<s>") to be generated.
  //   cg: Target computation graph.
  //   ig: Placeholder of the output inference graph.
  void decodeForInference(
      const dynet::expr::Expression & dec_init_h,
      const std::vector<dynet::expr::Expression> & atten_info,
      const unsigned bos_id,
      const unsigned eos_id,
      const unsigned max_length,
      dynet::ComputationGraph * cg,
      InferenceGraph * ig);

  // Constructs graph of the output loss function.
  //
  // Arguments:
  //   target_ids: Target word IDs for this step.
  //   dec_outputs: Decoder output distributions.
  //   losses: Placeholder of the loss expressions.
  void buildLossGraph(
      const std::vector<std::vector<unsigned>> & target_ids,
      const std::vector<dynet::expr::Expression> & dec_outputs,
      std::vector<dynet::expr::Expression> * losses);

  // Boost serialization interface.
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive & ar, const unsigned) {
    ar & encoder_;
    ar & enc2dec_;
    ar & dec2out_;
    ar & attention_;
    ar & rnn_dec_;
    ar & p_dec_lookup_;
  }

  boost::scoped_ptr<nmtkit::Encoder> encoder_;
  boost::scoped_ptr<nmtkit::MultilayerPerceptron> enc2dec_;
  boost::scoped_ptr<nmtkit::MultilayerPerceptron> dec2out_;
  boost::scoped_ptr<nmtkit::Attention> attention_;
  boost::scoped_ptr<dynet::LSTMBuilder> rnn_dec_;
  dynet::LookupParameter p_dec_lookup_;
};

}  // namespace nmtkit

NMTKIT_SERIALIZATION_DECL(nmtkit::EncoderDecoder);

#endif  // NMTKIT_ENCODER_DECODER_H_

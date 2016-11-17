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
#include <nmtkit/decoder.h>
#include <nmtkit/encoder.h>
#include <nmtkit/inference_graph.h>
#include <nmtkit/multilayer_perceptron.h>
#include <nmtkit/serialization_utils.h>
#include <nmtkit/predictor.h>

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
  //   encoder_type: Name of the encoder.
  //                 Available values:
  //                   "bidirectional": Bidirectional RNN.
  //                   "forward": Forward RNN.
  //                   "backward": Backward RNN.
  //   decoder_type: Name of the decoder.
  //                 Available values:
  //                   "default": Default RNN decoder.
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
      const std::string & encoder_type,
      const std::string & decoder_type,
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
  //   beam_width: Beam width.
  //   word_penalty: Word penalty.
  //   cg: Target computation graph.
  //   ig: Placeholder of the output inference graph.
  void infer(
      const std::vector<unsigned> & source_ids,
      const unsigned bos_id,
      const unsigned eos_id,
      const unsigned max_length,
      const unsigned beam_width,
      const float word_penalty,
      dynet::ComputationGraph * cg,
      InferenceGraph * ig);

private:
  // Constructs decoder graph for training.
  //
  // Arguments:
  //   seed: seed value of the decoder.
  //   target_ids: Target word IDs for this step.
  //   cg: Target computation graph.
  //
  // Returns:
  //   List of expression objects representing logit values.
  std::vector<dynet::expr::Expression> buildDecoderGraph(
      const dynet::expr::Expression & seed,
      const std::vector<std::vector<unsigned>> & target_ids,
      dynet::ComputationGraph * cg);

  // Generates output sequence using encoder results.
  //
  // Arguments:
  //   seed: seed value of the decoder.
  //   bos_id: "<s>" ID in the target language.
  //   eos_id: "</s>" ID in the target language.
  //   max_length: Maximum number of words (except "<s>") to be generated.
  //   beam_width: Beam width.
  //   word_penalty: Word penalty.
  //   cg: Target computation graph.
  //   ig: Placeholder of the output inference graph.
  void decodeForInference(
      const dynet::expr::Expression & seed,
      const unsigned bos_id,
      const unsigned eos_id,
      const unsigned max_length,
      const unsigned beam_width,
      const float word_penalty,
      dynet::ComputationGraph * cg,
      InferenceGraph * ig);

  // Boost serialization interface.
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive & ar, const unsigned) {
    ar & encoder_;
    ar & enc2dec_;
    ar & dec2logit_;
    ar & attention_;
    ar & decoder_;
    ar & predictor_;
  }

  boost::scoped_ptr<nmtkit::Encoder> encoder_;
  boost::scoped_ptr<nmtkit::MultilayerPerceptron> enc2dec_;
  boost::scoped_ptr<nmtkit::MultilayerPerceptron> dec2logit_;
  boost::scoped_ptr<nmtkit::Attention> attention_;
  boost::scoped_ptr<nmtkit::Decoder> decoder_;
  boost::scoped_ptr<nmtkit::Predictor> predictor_;
};

}  // namespace nmtkit

NMTKIT_SERIALIZATION_DECL(nmtkit::EncoderDecoder);

#endif  // NMTKIT_ENCODER_DECODER_H_

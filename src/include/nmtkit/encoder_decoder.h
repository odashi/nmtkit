#ifndef NMTKIT_ENCODER_DECODER_H_
#define NMTKIT_ENCODER_DECODER_H_

#include <string>
#include <vector>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/scoped_ptr.hpp>
#include <boost/serialization/shared_ptr.hpp>
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
  //   encoder: Pointer to the Encoder object.
  //   decoder: Pointer to the Decoder object.
  //   attention: Pointer to the Attention object.
  //   trg_vocab_size: Target vocabulary size.
  //   model: Model object for training.
  EncoderDecoder(
      boost::shared_ptr<Encoder> & encoder,
      boost::shared_ptr<Decoder> & decoder,
      boost::shared_ptr<Attention> & attention,
      unsigned trg_vocab_size,
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
    ar & decoder_;
    ar & attention_;
    ar & dec2logit_;
    ar & predictor_;
  }

  boost::shared_ptr<Encoder> encoder_;
  boost::shared_ptr<Decoder> decoder_;
  boost::shared_ptr<Attention> attention_;
  boost::scoped_ptr<MultilayerPerceptron> dec2logit_;
  boost::scoped_ptr<Predictor> predictor_;
};

}  // namespace nmtkit

NMTKIT_SERIALIZATION_DECL(nmtkit::EncoderDecoder);

#endif  // NMTKIT_ENCODER_DECODER_H_

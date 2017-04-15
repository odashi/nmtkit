#ifndef NMTKIT_BAHDANAU_DECODER_H_
#define NMTKIT_BAHDANAU_DECODER_H_

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/vector.hpp>
#include <dynet/lstm.h>
#include <dynet/model.h>
#include <nmtkit/decoder.h>
#include <nmtkit/multilayer_perceptron.h>
#include <nmtkit/serialization_utils.h>

namespace nmtkit {

// Decoder implementation proposed by Bahdanau et al., 2014.
// https://arxiv.org/abs/1409.0473
class BahdanauDecoder : public Decoder {
  BahdanauDecoder(const BahdanauDecoder &) = delete;
  BahdanauDecoder(BahdanauDecoder &&) = delete;
  BahdanauDecoder & operator=(const BahdanauDecoder &) = delete;
  BahdanauDecoder & operator=(BahdanauDecoder &&) = delete;

public:
  // Initializes an empty decoder.
  BahdanauDecoder() {}

  // Initializes a decoder.
  //
  // Arguments:
  //   num_layers: Depth of the decoder stack.
  //   vocab_size: Vocabulary size of the input sequences.
  //   in_embed_size: Number of units in the input embedding layer.
  //   out_embed_size: Number of units in the output embedding layer.
  //   hidden_size: Number of units in each RNN hidden layer.
  //   seed_size: Number of units in the seed vector.
  //   context_size: Number of units in the context vector.
  //   model: Model object for training.
  BahdanauDecoder(
      unsigned num_layers,
      unsigned vocab_size,
      unsigned in_embed_size,
      unsigned out_embed_size,
      unsigned hidden_size,
      unsigned seed_size,
      unsigned context_size,
      dynet::Model * model);

  ~BahdanauDecoder() override {}

  Decoder::State prepare(
      const std::vector<dynet::expr::Expression> & seed,
      const float dropout_ratio,
      dynet::ComputationGraph * cg) override;

  Decoder::State oneStep(
      const Decoder::State & state,
      const std::vector<unsigned> & input_ids,
      Attention * attention,
      dynet::ComputationGraph * cg,
      dynet::expr::Expression * atten_probs,
      dynet::expr::Expression * output) override;

  unsigned getOutputSize() const override { return out_embed_size_; }

private:
  // Boost serialization interface.
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive & ar, const unsigned) {
    ar & boost::serialization::base_object<Decoder>(*this);
    ar & num_layers_;
    ar & vocab_size_;
    ar & in_embed_size_;
    ar & out_embed_size_;
    ar & hidden_size_;
    ar & seed_size_;
    ar & context_size_;
    ar & enc2dec_;
    ar & dec2out_;
    ar & rnn_;
    ar & p_lookup_;
  }

  unsigned num_layers_;
  unsigned vocab_size_;
  unsigned in_embed_size_;
  unsigned out_embed_size_;
  unsigned hidden_size_;
  unsigned seed_size_;
  unsigned context_size_;
  std::vector<MultilayerPerceptron> enc2dec_;
  MultilayerPerceptron dec2out_;
  dynet::LSTMBuilder rnn_;
  dynet::LookupParameter p_lookup_;
};

}  // namespace nmtkit

NMTKIT_SERIALIZATION_DECL(nmtkit::BahdanauDecoder);

#endif  // NMTKIT_BAHDANAU_DECODER_H_

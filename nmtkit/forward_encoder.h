#ifndef NMTKIT_FORWARD_ENCODER_H_
#define NMTKIT_FORWARD_ENCODER_H_

#include <boost/serialization/base_object.hpp>
#include <dynet/lstm.h>
#include <dynet/model.h>
#include <nmtkit/encoder.h>
#include <nmtkit/serialization_utils.h>

namespace nmtkit {

// Forward encoder.
class ForwardEncoder : public Encoder {
  ForwardEncoder(const ForwardEncoder &) = delete;
  ForwardEncoder(ForwardEncoder &&) = delete;
  ForwardEncoder & operator=(const ForwardEncoder &) = delete;
  ForwardEncoder & operator=(ForwardEncoder &&) = delete;

public:
  // Initializes an empty encoder object.
  ForwardEncoder() {}

  // Initializes an encoder object.
  // Arguments:
  //   num_layers: Depth of the RNN stacks.
  //   vocab_size: Vocabulary size of the input sequences.
  //   embed_size: Number of units in the embedding layer.
  //   hidden_size: Number of units in each RNN hidden layer.
  //   dropout_rate: Dropout probability.
  //   model: Model object for training.
  ForwardEncoder(
      const unsigned num_layers,
      const unsigned vocab_size,
      const unsigned embed_size,
      const unsigned hidden_size,
      const float dropout_rate,
      dynet::Model * model);

  ~ForwardEncoder() override {}

  void prepare(
      dynet::ComputationGraph * cg,
      const bool is_training) override;
  std::vector<dynet::expr::Expression> compute(
      const std::vector<std::vector<unsigned>> & input_ids,
      dynet::ComputationGraph * cg,
      const bool is_training) override;

  std::vector<dynet::expr::Expression> getStates() const override;
  unsigned getOutputSize() const override { return hidden_size_; }
  unsigned getStateSize() const override { return hidden_size_; }

private:
  // Boost serialization interface.
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive & ar, const unsigned) {
    ar & boost::serialization::base_object<Encoder>(*this);
    ar & num_layers_;
    ar & vocab_size_;
    ar & embed_size_;
    ar & hidden_size_;
    ar & dropout_rate_;
    ar & rnn_;
    ar & p_lookup_;
  }

  unsigned num_layers_;
  unsigned vocab_size_;
  unsigned embed_size_;
  unsigned hidden_size_;
  float dropout_rate_;
  LSTM_MODULE rnn_;
  dynet::LookupParameter p_lookup_;
};

}  // namespace nmtkit

NMTKIT_SERIALIZATION_DECL(nmtkit::ForwardEncoder);

#endif  // NMTKIT_FORWARD_ENCODER_H_

#ifndef NMTKIT_BIDIRECTIONAL_ENCODER_H_
#define NMTKIT_BIDIRECTIONAL_ENCODER_H_

#include <boost/serialization/base_object.hpp>
#include <dynet/lstm.h>
#include <dynet/model.h>
#include <nmtkit/encoder.h>
#include <nmtkit/serialization_utils.h>

namespace nmtkit {

// Bidirectional encoder.
class BidirectionalEncoder : public Encoder {
  BidirectionalEncoder(const BidirectionalEncoder &) = delete;
  BidirectionalEncoder(BidirectionalEncoder &&) = delete;
  BidirectionalEncoder & operator=(const BidirectionalEncoder &) = delete;
  BidirectionalEncoder & operator=(BidirectionalEncoder &&) = delete;

public:
  // Initializes an empty encoder object.
  BidirectionalEncoder() {}

  // Initializes encoder object.
  // Arguments:
  //   num_layers: Depth of the RNN stacks.
  //   vocab_size: Vocabulary size of the input sequences.
  //   embed_size: Number of units in the embedding layer.
  //   hidden_size: Number of units in each RNN hidden layer.
  //   model: Model object for training.
  BidirectionalEncoder(
      unsigned num_layers,
      unsigned vocab_size,
      unsigned embed_size,
      unsigned hidden_size,
      dynet::Model * model);

  ~BidirectionalEncoder() override {}

  void prepare(dynet::ComputationGraph * cg) override;
  std::vector<dynet::expr::Expression> compute(
      const std::vector<std::vector<unsigned>> & input_ids,
      dynet::ComputationGraph * cg) override;

  std::vector<dynet::expr::Expression> getStates() const override;
  unsigned getOutputSize() const override { return 2 * hidden_size_; }
  unsigned getStateSize() const override { return 2 * hidden_size_; }

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
    ar & rnn_fw_;
    ar & rnn_bw_;
    ar & p_lookup_;
  }

  unsigned num_layers_;
  unsigned vocab_size_;
  unsigned embed_size_;
  unsigned hidden_size_;
  dynet::LSTMBuilder rnn_fw_;
  dynet::LSTMBuilder rnn_bw_;
  dynet::LookupParameter p_lookup_;
};

}  // namespace nmtkit

NMTKIT_SERIALIZATION_DECL(nmtkit::BidirectionalEncoder);

#endif  // NMTKIT_BIDIRECTIONAL_ENCODER_H_

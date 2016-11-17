#ifndef NMTKIT_DECODER_H_
#define NMTKIT_DECODER_H_

#include <vector>
#include <boost/serialization/access.hpp>
#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <nmtkit/attention.h>
#include <nmtkit/serialization_utils.h>

namespace nmtkit {

// Abstract class for the decoder implementations.
class Decoder {
  Decoder(const Decoder &) = delete;
  Decoder(Decoder &&) = delete;
  Decoder & operator=(const Decoder &) = delete;
  Decoder & operator=(Decoder &&) = delete;

public:
  Decoder() {}
  virtual ~Decoder() {}

  // Prepares internal parameters.
  //
  // Arguments:
  //   seed: Seed values of initial states, e.g., final encoder states.
  //   cg: Target computation graph.
  //
  // Returns:
  //   Initial states of the decoder.
  virtual std::vector<dynet::expr::Expression> prepare(
      const dynet::expr::Expression & seed,
      dynet::ComputationGraph * cg) = 0;

  // Proceeds one decoding step.
  //
  // Arguments:
  //   states: Previous states.
  //   input_ids: List of input symbols in the current step.
  //   attention: Attention object.
  //   cg: Target computation graph.
  //   atten_probs: Placeholder of the attention probability vector. If the
  //                value is nullptr, this argument would be ignored.
  //   output: Placeholder of the output embedding. If the value is nullptr,
  //           this argument would be ignored.
  //
  // Returns:
  //   Next states of the decoder.
  virtual std::vector<dynet::expr::Expression> oneStep(
      const std::vector<dynet::expr::Expression> & states,
      const std::vector<unsigned> & input_ids,
      Attention * attention,
      dynet::ComputationGraph * cg,
      dynet::expr::Expression * atten_probs,
      dynet::expr::Expression * output) = 0;

  // Returns the number of units in the output embedding.
  virtual unsigned getOutputSize() const = 0;

private:
  // Boost serialization interface.
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive & ar, const unsigned) {}
};

}  // namespace nmtkit

NMTKIT_SERIALIZATION_DECL(nmtkit::Decoder);

#endif  // NMTKIT_DECODER_H_

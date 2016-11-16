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
  //   cg; Target computation graph.
  //
  // Returns:
  //   List of parameters.
  virtual std::vector<dynet::expr::Expression> prepare(
      dynet::ComputationGraph * cg) = 0;

  // Obtains initial decoder states.
  //
  // Arguments:
  //   seed: Seed values of initial states, e.g., final encoder states.
  //   cg: Target computation graph.
  //
  // Returns:
  //   Initial states of the decoder.
  virtual std::vector<dynet::expr::Expression> initialize(
      const dynet::expr::Expression & seed,
      dynet::ComputationGraph * cg) = 0;

  // Proceeds one decoding step.
  //
  // Arguments:
  //   input_ids: List of input symbols in the current step.
  //   states: Previous decoder states.
  //   dec_params: List of decoder parameters returned by prepare().
  //   atten_params: List of attention parameters returned by
  //                 Attention::prepare().
  //   atten: Attention object.
  //   cg: Target computation graph.
  //   output: Placeholder of the output embedding. This argument would be
  //           ignored if the value is nullptr.
  //
  // Returns:
  //   Next states of the decoder.
  virtual std::vector<dynet::expr::Expression> oneStep(
      const std::vector<unsigned> & input_ids,
      const std::vector<dynet::expr::Expression> & states,
      const std::vector<dynet::expr::Expression> & dec_params,
      const std::vector<dynet::expr::Expression> & atten_params,
      Attention * attention,
      dynet::ComputationGraph * cg,
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

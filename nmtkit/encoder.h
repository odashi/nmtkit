#ifndef NMTKIT_ENCODER_H_
#define NMTKIT_ENCODER_H_

#include <vector>
#include <boost/serialization/access.hpp>
#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <nmtkit/serialization_utils.h>

namespace nmtkit {

// Abstract class for the encoder implementation.
class Encoder {
  Encoder(const Encoder &) = delete;
  Encoder(Encoder &&) = delete;
  Encoder & operator=(const Encoder &) = delete;
  Encoder & operator=(Encoder &&) = delete;

public:
  Encoder() {}
  virtual ~Encoder() {}

  // Initializes internal states.
  //
  // Arguments:
  //   cg: Computation graph.
  //   is_training: true when training, false otherwise.
  virtual void prepare(
      dynet::ComputationGraph * cg,
      const bool is_training) = 0;

  // Calculates outputs of all inputs.
  //
  // Arguments:
  //   input_ids: List of input symbols as following format:
  //     { { sample_1[0], sample_2[0], ..., sample_n[0] },
  //       { sample_1[1], sample_2[1], ..., sample_n[1] },
  //       ...,
  //       { sample_1[m], sample_2[m], ..., sample_n[m] } }
  //   cg: Computation graph.
  //   is_training: true when training, false otherwise.
  //
  // Returns:
  //   List of expressions representing the output of each input.
  virtual std::vector<dynet::expr::Expression> compute(
      const std::vector<std::vector<unsigned>> & input_ids,
      dynet::ComputationGraph * cg,
      const bool is_training) = 0;

  // Retrieves the list of final states.
  virtual std::vector<dynet::expr::Expression> getStates() const = 0;

  // Retrieves the number of units in each output node.
  virtual unsigned getOutputSize() const = 0;

  // Retrieves the number of units in the final state node.
  virtual unsigned getStateSize() const = 0;

private:
  // Boost serialization interface.
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive & ar, const unsigned) {}
};

}  // namespace nmtkit

NMTKIT_SERIALIZATION_DECL(nmtkit::Encoder);

#endif // NMTKIT_ENCODER_H_

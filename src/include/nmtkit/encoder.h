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

  // Constructs computation graph.
  // Arguments:
  //   input_ids: List of input symbols as following format:
  //     { { sample_1[0], sample_2[0], ..., sample_n[0] },
  //       { sample_1[1], sample_2[1], ..., sample_n[1] },
  //       ...,
  //       { sample_1[m], sample_2[m], ..., sample_n[m] } }
  //   cg: Target computation graph.
  //   output_states: Placeholder of the output hidden states for each inputs.
  //                  This argument could be ignored by passing nullptr.
  //   final_state: Placeholder of the last states of the inner network.
  //                This argument could be ignored by passing nullptr.
  virtual void build(
      const std::vector<std::vector<unsigned>> & input_ids,
      dynet::ComputationGraph * cg,
      std::vector<dynet::expr::Expression> * output_states,
      dynet::expr::Expression * final_state) = 0;

  // Retrieves the number of units in each state node.
  // Returns:
  //   Nubmer of units.
  virtual unsigned getStateSize() const = 0;

  // Retrieves the number of units in the final state node.
  virtual unsigned getFinalStateSize() const = 0;

private:
  // Boost serialization interface.
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive & ar, const unsigned) {}
};

}  // namespace nmtkit

NMTKIT_SERIALIZATION_DECL(nmtkit::Encoder);

#endif // NMTKIT_ENCODER_H_


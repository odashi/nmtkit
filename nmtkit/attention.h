#ifndef NMTKIT_ATTENTION_H_
#define NMTKIT_ATTENTION_H_

#include <vector>
#include <boost/serialization/access.hpp>
#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <nmtkit/serialization_utils.h>

namespace nmtkit {

// Abstract class for attention implementations.
class Attention {
  Attention(const Attention &) = delete;
  Attention(Attention &&) = delete;
  Attention & operator=(const Attention &) = delete;
  Attention & operator=(Attention &&) = delete;

public:
  Attention() {}
  virtual ~Attention() {}

  // Converts input memory arrays into precomputed values.
  //
  // Arguments:
  //   memories: List of input memory arrays. All expression should be a vector
  //             with same number of units.
  //   cg: Computation graph.
  virtual void prepare(
      const std::vector<dynet::expr::Expression> & memories,
      dynet::ComputationGraph * cg) = 0;

  // Calculate the attention distribution and the context vector.
  //
  // Arguments:
  //   controller: An input vector expression to compute the attention
  //               distribution.
  //
  // Returns:
  //   2 expression objects:
  //     [0]: Expression object representing the attention probability vector.
  //     [1]: Expression object representing the context vector.
  virtual std::vector<dynet::expr::Expression> compute(
      const dynet::expr::Expression & controller) = 0;

private:
  // Boost serialization interface.
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive & ar, const unsigned) {}
};

}  // namespace nmtkit

NMTKIT_SERIALIZATION_DECL(nmtkit::Attention);

#endif  // NMTKIT_ATTENTION_H_

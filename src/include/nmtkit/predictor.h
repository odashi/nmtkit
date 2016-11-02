#ifndef NMTKIT_PREDICTOR_H_
#define NMTKIT_PREDICTOR_H_

#include <vector>
#include <boost/serialization/access.hpp>
#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <nmtkit/serialization_utils.h>

namespace nmtkit {

// Abstract class of the word predictor.
class Predictor {
  Predictor(const Predictor &) = delete;
  Predictor(Predictor &&) = delete;
  Predictor & operator=(const Predictor &) = delete;
  Predictor & operator=(Predictor &&) = delete;

public:
  // Output candidates of the predictor.
  struct Result {
    unsigned word_id;
    float log_prob;
  };

  Predictor() {}
  virtual ~Predictor() {}

  // Calculates the loss value of given logits.
  //
  // Arguments:
  //   target_ids: Target word IDs for all outputs.
  //   logits: List of the expression object which represents logit values
  //                for all outputs.
  //
  // Returns:
  //   Expression object of the total loss value.
  virtual dynet::expr::Expression computeLoss(
      const std::vector<std::vector<unsigned>> & target_ids,
      const std::vector<dynet::expr::Expression> & logits) = 0;

  // Predicts k-best words using given vector.
  //
  // Arguments:
  //   logit: Expression object which describes logit values of one output
  //          layer.
  //   num_results: Number of results to be obtained.
  //   cg: Target computation graph.
  //
  // Returns:
  //   List of top-k candidates. The order of elements in the output vector
  //   would be sorted by the decsending order according to their probabilities.
  virtual std::vector<Result> predictKBest(
      const dynet::expr::Expression & logit,
      unsigned num_results,
      dynet::ComputationGraph * cg) = 0;

  // Obtains log probabilities of specific word IDs.
  //
  // Arguments:
  //   logit: Expression object which describes logit values of one output
  //          layer.
  //   word_ids: Target word IDs.
  //   cg: Target computation graph.
  //
  // Returns:
  //   List of candidates. The order of outputs would be similar to that of
  //   word_ids.
  virtual std::vector<Result> predictByIDs(
      const dynet::expr::Expression & logit,
      const std::vector<unsigned> word_ids,
      dynet::ComputationGraph * cg) = 0;

private:
  // Boost serialization interface.
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive & ar, const unsigned) {}
};

}  // namespace nmtkit

NMTKIT_SERIALIZATION_DECL(nmtkit::Predictor);

#endif  // NMTKIT_PREDICTOR_H_

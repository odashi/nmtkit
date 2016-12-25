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

  // Calculates the loss value of given scores.
  //
  // Arguments:
  //   target_ids: Target word IDs for all outputs.
  //   scores: List of the expression object which represents score values
  //                for all outputs.
  //   cg: Target computation graph.
  //
  // Returns:
  //   Expression object of the total loss value.
  virtual dynet::expr::Expression computeLoss(
      const std::vector<std::vector<unsigned>> & target_ids,
      const std::vector<dynet::expr::Expression> & scores,
      dynet::ComputationGraph * cg) = 0;

  // Predicts k-best words using given vector.
  //
  // Arguments:
  //   score: Expression object which describes score values of one output
  //          layer.
  //   num_results: Number of results to be obtained.
  //   cg: Target computation graph.
  //
  // Returns:
  //   List of top-k candidates. The order of elements in the output vector
  //   would be sorted by the decsending order according to their probabilities.
  virtual std::vector<Result> predictKBest(
      const dynet::expr::Expression & score,
      unsigned num_results,
      dynet::ComputationGraph * cg) = 0;

  // Obtains log probabilities of specific word IDs.
  //
  // Arguments:
  //   score: Expression object which describes score values of one output
  //          layer.
  //   word_ids: Target word IDs.
  //   cg: Target computation graph.
  //
  // Returns:
  //   List of candidates. The order of outputs would be similar to that of
  //   word_ids.
  virtual std::vector<Result> predictByIDs(
      const dynet::expr::Expression & score,
      const std::vector<unsigned> word_ids,
      dynet::ComputationGraph * cg) = 0;

  // Returns size of the input layer.
  virtual unsigned getScoreSize() const = 0;

private:
  // Boost serialization interface.
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive & ar, const unsigned) {}
};

}  // namespace nmtkit

NMTKIT_SERIALIZATION_DECL(nmtkit::Predictor);

#endif  // NMTKIT_PREDICTOR_H_

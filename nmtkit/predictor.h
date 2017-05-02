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

  // Prepares the predictor.
  //
  // Arguments:
  //   cg: Computation graph.
  //   is_training: true when training, false otherwise.
  virtual void prepare(
      dynet::ComputationGraph * cg,
      const bool is_training) = 0;

  // Calculates loss values.
  //
  // Arguments:
  //   input: Expression object representing input vector.
  //   target_ids: Target word IDs.
  //   cg: Computation graph.
  //   is_training: true when training, false otherwise.
  //
  // Returns:
  //   Expression object representing loss values.
  virtual dynet::expr::Expression computeLoss(
      const dynet::expr::Expression & input,
      const std::vector<unsigned> & target_ids,
      dynet::ComputationGraph * cg,
      const bool is_training) = 0;

  // Predicts k-best words.
  //
  // Arguments:
  //   input: Expression object representing input vector.
  //   num_results: Number of results to be obtained.
  //   cg: Computation graph.
  //
  // Returns:
  //   List of top-k candidates. The order of elements in the output vector
  //   would be sorted by the decsending order about their probabilities.
  virtual std::vector<Result> predictKBest(
      const dynet::expr::Expression & input,
      const unsigned num_results,
      dynet::ComputationGraph * cg) = 0;

  // Obtains log probabilities of specific word IDs.
  //
  // Arguments:
  //   input: Expression object representing input vector.
  //   word_ids: Target word IDs.
  //   cg: Computation graph.
  //
  // Returns:
  //   List of candidates. The order of outputs would be similar to that of
  //   word_ids.
  virtual std::vector<Result> predictByIDs(
      const dynet::expr::Expression & input,
      const std::vector<unsigned> word_ids,
      dynet::ComputationGraph * cg) = 0;

  // Sample a candidate sentence from a whole translation probability.
  //
  // Arguments:
  //   input: Expression object representing input vector.
  //   cg: Computation graph.
  //
  // Returns:
  //   Sampled candidate.
  virtual Result sample(
      const dynet::expr::Expression & input,
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

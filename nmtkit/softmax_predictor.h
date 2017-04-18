#ifndef NMTKIT_SOFTMAX_PREDICTOR_H_
#define NMTKIT_SOFTMAX_PREDICTOR_H_

#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <nmtkit/predictor.h>
#include <nmtkit/multilayer_perceptron.h>
#include <nmtkit/serialization_utils.h>

namespace nmtkit {

// Word predictor using softmax outputs.
class SoftmaxPredictor : public Predictor {
  SoftmaxPredictor(const SoftmaxPredictor &) = delete;
  SoftmaxPredictor(SoftmaxPredictor &&) = delete;
  SoftmaxPredictor & operator=(const SoftmaxPredictor &) = delete;
  SoftmaxPredictor & operator=(SoftmaxPredictor &&) = delete;

public:
  // Initializes an empty predictor.
  SoftmaxPredictor() {}

  // Initializes the predictor.
  //
  // Arguments:
  //   input_size: Number of units in the input vector.
  //   vocab_size: Vocabulary size of the target language.
  //   model: Model object for training.
  SoftmaxPredictor(
      const unsigned input_size,
      const unsigned vocab_size,
      dynet::Model * model);

  ~SoftmaxPredictor() override {}

  void prepare(
      dynet::ComputationGraph * cg,
      const bool is_training) override;

  dynet::expr::Expression computeLoss(
      const dynet::expr::Expression & input,
      const std::vector<unsigned> & target_ids,
      dynet::ComputationGraph * cg,
      const bool is_training) override;

  std::vector<Predictor::Result> predictKBest(
      const dynet::expr::Expression & input,
      const unsigned num_results,
      dynet::ComputationGraph * cg) override;

  std::vector<Predictor::Result> predictByIDs(
      const dynet::expr::Expression & input,
      const std::vector<unsigned> word_ids,
      dynet::ComputationGraph * cg) override;

private:
  // Boost serialization interface.
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive & ar, const unsigned) {
    ar & boost::serialization::base_object<Predictor>(*this);
    ar & vocab_size_;
    ar & converter_;
  }

  unsigned vocab_size_;
  MultilayerPerceptron converter_;
};

}  // namespace nmtkit

NMTKIT_SERIALIZATION_DECL(nmtkit::SoftmaxPredictor);

#endif  // NMTKIT_SOFTMAX_PREDICTOR_H_

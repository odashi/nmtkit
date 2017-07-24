#ifndef NMTKIT_SEPARATED_SOFTMAX_PREDICTOR_H_
#define NMTKIT_SEPARATED_SOFTMAX_PREDICTOR_H_

#include <memory>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <nmtkit/multilayer_perceptron.h>
#include <nmtkit/predictor.h>
#include <nmtkit/serialization_utils.h>

namespace nmtkit {

// Word predictor using two-way softmax prediction.
class SeparatedSoftmaxPredictor : public Predictor {
  SeparatedSoftmaxPredictor(const SeparatedSoftmaxPredictor &) = delete;
  SeparatedSoftmaxPredictor(SeparatedSoftmaxPredictor &&) = delete;
  SeparatedSoftmaxPredictor & operator=(const SeparatedSoftmaxPredictor &) = delete;
  SeparatedSoftmaxPredictor & operator=(SeparatedSoftmaxPredictor &&) = delete;

public:
  // Initializes an empty predictor.
  SeparatedSoftmaxPredictor() {}

  // Initializes the predictor.
  //
  // Arguments:
  //   input_size: Number of units in the input vector.
  //   vocab_size: Output vocabulary size.
  //   first_size: Size of the vocabulary of frequent words and <unk>.
  //   model: Model object for training.
  SeparatedSoftmaxPredictor(
      const unsigned input_size,
      const unsigned vocab_size,
      const unsigned first_size,
      dynet::Model * model);

  ~SeparatedSoftmaxPredictor() override {}

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

  Predictor::Result sample(
      const dynet::expr::Expression & input,
      dynet::ComputationGraph * cg) override;

private:
  // Boost serialization interface.
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive & ar, const unsigned) {
    ar & boost::serialization::base_object<Predictor>(*this);
    ar & vocab_size_;
    ar & first_size_;
    ar & first_converter_;
    ar & second_converter_;
  }

  unsigned vocab_size_;
  unsigned first_size_;
  MultilayerPerceptron first_converter_;
  MultilayerPerceptron second_converter_;
};

}  // namespace nmtkit

NMTKIT_SERIALIZATION_DECL(nmtkit::SeparatedSoftmaxPredictor);

#endif  // NMTKIT_SEPARATED_SOFTMAX_PREDICTOR_H_

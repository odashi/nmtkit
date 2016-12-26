#ifndef NMTKIT_BINARY_CODE_PREDICTOR_H_
#define NMTKIT_BINARY_CODE_PREDICTOR_H_

#include <string>
#include <boost/shared_ptr.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <nmtkit/binary_code.h>
#include <nmtkit/multilayer_perceptron.h>
#include <nmtkit/predictor.h>
#include <nmtkit/serialization_utils.h>

namespace nmtkit {

// Word predictor using binary code outputs.
class BinaryCodePredictor : public Predictor {
  BinaryCodePredictor(const BinaryCodePredictor &) = delete;
  BinaryCodePredictor(BinaryCodePredictor &&) = delete;
  BinaryCodePredictor & operator=(const BinaryCodePredictor &) = delete;
  BinaryCodePredictor & operator=(BinaryCodePredictor &&) = delete;

public:
  // Initializes an empty predictor.
  BinaryCodePredictor() {}

  // Initializes the predictor.
  //
  // Arguments:
  //   input_size: Number of units in the input vector.
  //   bc: Pointer to a BinaryCode object.
  //   model: Model object for training.
  BinaryCodePredictor(
      const unsigned input_size,
      boost::shared_ptr<BinaryCode> & bc,
      dynet::Model * model);

  ~BinaryCodePredictor() override {}

  void prepare(dynet::ComputationGraph * cg) override;

  dynet::expr::Expression computeLoss(
      const dynet::expr::Expression & input,
      const std::vector<unsigned> & target_ids,
      dynet::ComputationGraph * cg) override;

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
    ar & bc_;
    ar & converter_;
  }

  boost::shared_ptr<BinaryCode> bc_;
  MultilayerPerceptron converter_;
};

}  // namespace nmtkit

NMTKIT_SERIALIZATION_DECL(nmtkit::BinaryCodePredictor);

#endif  // NMTKIT_BINARY_CODE_PREDICTOR_H_

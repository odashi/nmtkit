#ifndef NMTKIT_SOFTMAX_PREDICTOR_H_
#define NMTKIT_SOFTMAX_PREDICTOR_H_

#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <nmtkit/predictor.h>
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
  //   vocab_size: Vocabulary size of the target language.
  SoftmaxPredictor(unsigned vocab_size);

  ~SoftmaxPredictor() override {}

  dynet::expr::Expression computeLoss(
      const std::vector<std::vector<unsigned>> & target_ids,
      const std::vector<dynet::expr::Expression> & logits) override;

  std::vector<PredictorResult> predictKBest(
      const dynet::expr::Expression & logit,
      unsigned num_results,
      dynet::ComputationGraph * cg) override;

  std::vector<PredictorResult> predictByIDs(
      const dynet::expr::Expression & logit,
      const std::vector<unsigned> word_ids,
      dynet::ComputationGraph * cg) override;

private:
  // Boost serialization interface.
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive & ar, const unsigned) {
    ar & boost::serialization::base_object<Predictor>(*this);
    ar & vocab_size_;
  }

  unsigned vocab_size_;
};

}  // namespace nmtkit

NMTKIT_SERIALIZATION_DECL(nmtkit::SoftmaxPredictor);

#endif  // NMTKIT_SOFTMAX_PREDICTOR_H_

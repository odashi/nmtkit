#ifndef NMTKIT_BILINEAR_ATTENTION_H_
#define NMTKIT_BILINEAR_ATTENTION_H_

#include <boost/serialization/base_object.hpp>
#include <dynet/model.h>
#include <nmtkit/attention.h>
#include <nmtkit/serialization_utils.h>

namespace nmtkit {

// Bilinear attention.
// score = mem^T * W * ctrl
class BilinearAttention : public Attention {
  BilinearAttention(const BilinearAttention &) = delete;
  BilinearAttention(BilinearAttention &&) = delete;
  BilinearAttention & operator=(const BilinearAttention &) = delete;
  BilinearAttention & operator=(BilinearAttention &&) = delete;

public:
  // Initializes an empty attention object.
  BilinearAttention() {}

  // Initializes attention object.
  //
  // Arguments:
  //   memory_size: Number of units in each memory input.
  //   controller_size: Number of units in the controller input.
  //   model: Model object for training.
  BilinearAttention(
      unsigned memory_size,
      unsigned controller_size,
      dynet::Model * model);

  ~BilinearAttention() override {}

  void prepare(
      const std::vector<dynet::expr::Expression> & memories,
      dynet::ComputationGraph * cg) override;

  std::vector<dynet::expr::Expression> compute(
      const dynet::expr::Expression & controller,
      dynet::ComputationGraph * cg) override;

private:
  // Boost serialization interface.
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive & ar, const unsigned) {
    ar & boost::serialization::base_object<Attention>(*this);
    ar & p_interaction_;
  }

  dynet::Parameter p_interaction_;
  dynet::expr::Expression i_concat_mem_;
  dynet::expr::Expression i_converted_mem_;
};

}  // namespace nmtkit

NMTKIT_SERIALIZATION_DECL(nmtkit::BilinearAttention);

#endif  // NMTKIT_BILINEAR_ATTENTION_H_

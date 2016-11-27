#include "config.h"

#include <nmtkit/multilayer_perceptron.h>

#include <nmtkit/exception.h>

using namespace std;

namespace DE = dynet::expr;

namespace nmtkit {

MultilayerPerceptron::MultilayerPerceptron(
    const vector<unsigned> & spec,
    dynet::Model * model)
: spec_(spec) {
  NMTKIT_CHECK(spec.size() >= 2, "Required at least 2 numbers in spec.");
  for (const unsigned units : spec) {
    NMTKIT_CHECK(units > 0, "All numbers in spec should be greater than 0.");
  }
  for (unsigned i = 0; i < spec_.size() - 1; ++i) {
    const unsigned in_size = spec_[i];
    const unsigned out_size = spec_[i + 1];
    w_.emplace_back(model->add_parameters({out_size, in_size}));
    b_.emplace_back(model->add_parameters({out_size}));
  }
}

void MultilayerPerceptron::prepare(dynet::ComputationGraph * cg) {
  i_w_.clear();
  i_b_.clear();
  for (unsigned i = 0; i < w_.size(); ++i) {
    i_w_.emplace_back(DE::parameter(*cg, w_[i]));
    i_b_.emplace_back(DE::parameter(*cg, b_[i]));
  }
}

DE::Expression MultilayerPerceptron::compute(const DE::Expression & input) {
  DE::Expression h = input;
  for (unsigned i = 0; i < w_.size(); ++i) {
    h = DE::affine_transform({i_b_[i], i_w_[i], h});
    if (i < w_.size() - 1) {
      // Hidden layers except the last one are nonliniarized by ReLU.
      h = DE::rectify(h);
    }
  }
  return h;
}

}  // namespace nmtkit

NMTKIT_SERIALIZATION_IMPL(nmtkit::MultilayerPerceptron);

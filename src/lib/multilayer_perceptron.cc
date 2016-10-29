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

vector<DE::Expression> MultilayerPerceptron::prepare(
    dynet::ComputationGraph * cg) {
  vector<DE::Expression> params;
  // Note: params[2*i]: i-th matrix
  //       params[2*i+1]: i-th bias
  for (unsigned i = 0; i < w_.size(); ++i) {
    params.emplace_back(DE::parameter(*cg, w_[i]));
    params.emplace_back(DE::parameter(*cg, b_[i]));
  }
  return params;
}

DE::Expression MultilayerPerceptron::compute(
    const vector<DE::Expression> & precomputed,
    const DE::Expression & input,
    dynet::ComputationGraph * cg) {
  vector<DE::Expression> hidden {input};
  for (unsigned i = 0; i < w_.size(); ++i) {
    const DE::Expression & w = precomputed[2 * i];
    const DE::Expression & b = precomputed[2 * i + 1];
    DE::Expression u = DE::affine_transform({b, w, hidden.back()});
    if (i == w_.size() - 1) {
      // Last (output) layer does not be nonliniarized.
      hidden.emplace_back(u);
    } else {
      // Hidden layers are nonliniarized by ReLU.
      hidden.emplace_back(DE::rectify(u));
    }
  }
  return hidden.back();
}

}  // namespace nmtkit

NMTKIT_SERIALIZATION_IMPL(nmtkit::MultilayerPerceptron);

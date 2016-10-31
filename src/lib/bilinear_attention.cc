#include "config.h"

#include <nmtkit/bilinear_attention.h>

#include <utility>
#include <nmtkit/exception.h>

using namespace std;

namespace DE = dynet::expr;

namespace nmtkit {

BilinearAttention::BilinearAttention(
    unsigned memory_size,
    unsigned controller_size,
    dynet::Model * model) {
  NMTKIT_CHECK(
      memory_size > 0, "memory_size should be greater than 0.");
  NMTKIT_CHECK(
      controller_size > 0, "controller_size should be greater than 0.");
  
  p_interaction_ = model->add_parameters({controller_size, memory_size});
}

vector<DE::Expression> BilinearAttention::prepare(
    const vector<DE::Expression> & memories,
    dynet::ComputationGraph * cg) {
  // Concatenated memory matrix.
  // Shape: {memory_size, seq_length}
  DE::Expression concat_mem = DE::concatenate_cols(memories);

  // Interaction coefficients between the memory and the controller.
  // Shape: {controller_size, memory_size}
  DE::Expression interaction = DE::parameter(*cg, p_interaction_);

  // Precomputes conversion of the memory matrix.
  // Shape: {seq_length, controller_size}
  DE::Expression converted_mem = DE::transpose(interaction * concat_mem);

  return {concat_mem, converted_mem};
}

void BilinearAttention::compute(
    const vector<DE::Expression> & precomputed,
    const DE::Expression & controller,
    dynet::ComputationGraph * cg,
    DE::Expression * atten_probs,
    DE::Expression * context) {
  NMTKIT_CHECK_EQ(
      2, precomputed.size(), "Invalid number of precomputed values.");

  // Aliases
  const DE::Expression & concat_mem = precomputed[0];
  const DE::Expression & converted_mem = precomputed[1];

  // Computes attention.
  // Shape: {seq_length, 1}
  DE::Expression atten_probs_inner = DE::softmax(converted_mem * controller);

  // Computes the context vector.
  // Shape: {memory_size, 1}
  DE::Expression context_inner = concat_mem * atten_probs_inner;

  // Copies results.
  if (atten_probs != nullptr) {
    *atten_probs = std::move(atten_probs_inner);
  }
  if (context != nullptr) {
    *context = std::move(context_inner);
  }
}

}  // namespace nmtkit

NMTKIT_SERIALIZATION_IMPL(nmtkit::BilinearAttention);

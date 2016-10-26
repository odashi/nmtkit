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
  
  p_interaction_ = model->add_parameters({memory_size, controller_size});
}

vector<DE::Expression> BilinearAttention::prepare(
    const vector<DE::Expression> & memories,
    dynet::ComputationGraph * cg) {
  // Concatenated memory matrix.
  // Shape: {seq_length, memory_size}
  DE::Expression concat_mem = DE::transpose(DE::concatenate_cols(memories));

  // Interaction coefficients between the memory and the controller.
  // Shape: {memory_size, controller_size}
  DE::Expression interaction = DE::parameter(*cg, p_interaction_);

  return {concat_mem, interaction};
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
  const DE::Expression & interaction = precomputed[1];

  // Computes transformation only using controller first to avoid calculation
  // Amount.
  // Shape: {memory_size, 1}
  DE::Expression converted_ctrl = interaction * controller;

  // Computes attention.
  // Shape: {seq_length, 1}
  DE::Expression atten_probs_inner = DE::softmax(concat_mem * converted_ctrl);

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

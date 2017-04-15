#include <config.h>

#include <nmtkit/bilinear_attention.h>

#include <utility>
#include <nmtkit/exception.h>

using std::vector;

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

void BilinearAttention::prepare(
    const vector<DE::Expression> & memories,
    dynet::ComputationGraph * cg) {
  // Concatenated memory matrix.
  // Shape: {memory_size, seq_length}
  i_concat_mem_ = DE::concatenate_cols(memories);

  // Interaction coefficients between the memory and the controller.
  // Shape: {controller_size, memory_size}
  DE::Expression interaction = DE::parameter(*cg, p_interaction_);

  // Precomputes conversion of the memory matrix.
  // Shape: {seq_length, controller_size}
  i_converted_mem_ = DE::transpose(interaction * i_concat_mem_);
}

vector<DE::Expression> BilinearAttention::compute(
    const DE::Expression & controller) {
  // Computes attention.
  // Shape: {seq_length, 1}
  DE::Expression atten_probs_inner = DE::softmax(i_converted_mem_ * controller);

  // Computes the context vector.
  // Shape: {memory_size, 1}
  DE::Expression context_inner = i_concat_mem_ * atten_probs_inner;

  return {atten_probs_inner, context_inner};
}

}  // namespace nmtkit

NMTKIT_SERIALIZATION_IMPL(nmtkit::BilinearAttention);

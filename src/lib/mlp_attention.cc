#include <nmtkit/mlp_attention.h>

#include <utility>
#include <nmtkit/exception.h>

using namespace std;

namespace DE = dynet::expr;

namespace nmtkit {

MLPAttention::MLPAttention(
    unsigned memory_size,
    unsigned controller_size,
    unsigned hidden_size,
    dynet::Model * model) {
  NMTKIT_CHECK(
      memory_size > 0, "memory_size should be greater than 0.");
  NMTKIT_CHECK(
      controller_size > 0, "controller_size should be greater than 0.");
  NMTKIT_CHECK(
      hidden_size > 0, "hidden_size should be greater than 0.");
  
  p_mem2h_ = model->add_parameters({hidden_size, memory_size});
  p_ctrl2h_ = model->add_parameters({hidden_size, controller_size});
  p_h2logit_ = model->add_parameters({1, hidden_size});
}

vector<DE::Expression> MLPAttention::prepare(
    const vector<DE::Expression> & memories,
    dynet::ComputationGraph * cg) {
  // Concatenated memory matrix.
  // Shape: {memory_size, seq_length}
  DE::Expression concat_mem = DE::concatenate_cols(memories);

  // Dimention broadcasting matrix for the controller.
  // Shape: {1, seq_length}
  const unsigned seq_length = memories.size();
  DE::Expression bcast = DE::input(
      *cg, {1, seq_length}, vector<float>(seq_length, 1.0f));

  // Parameters
  DE::Expression mem2h = DE::parameter(*cg, p_mem2h_);
  DE::Expression ctrl2h = DE::parameter(*cg, p_ctrl2h_);
  DE::Expression h2logit = DE::parameter(*cg, p_h2logit_);

  // Precomputes memory -> hidden mapping.
  // Shape: {hidden_size, seq_length}
  DE::Expression h_mem = mem2h * concat_mem;

  // Returns 2 precomputed inputs and 2 params
  return {concat_mem, h_mem, bcast, ctrl2h, h2logit};
}

void MLPAttention::compute(
    const vector<DE::Expression> & precomputed,
    const DE::Expression & controller,
    dynet::ComputationGraph * cg,
    DE::Expression * atten_probs,
    DE::Expression * context) {
  NMTKIT_CHECK_EQ(
      5, precomputed.size(), "Invalid number of precomputed values.");

  // Aliases
  const DE::Expression & concat_mem = precomputed[0];
  const DE::Expression & h_mem = precomputed[1];
  const DE::Expression & bcast = precomputed[2];
  const DE::Expression & ctrl2h = precomputed[3];
  const DE::Expression & h2logit = precomputed[4];

  // Computes the attention distribution.
  // Shape: {hidden_size, 1}
  DE::Expression h_ctrl = ctrl2h * controller;
  // Shape: {hidden_size, seq_length}
  DE::Expression h = DE::tanh(DE::affine_transform({h_mem, h_ctrl, bcast}));
  // Shape: {seq_length, 1}
  DE::Expression atten_probs_inner = DE::softmax(DE::transpose(h2logit * h));

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

NMTKIT_SERIALIZATION_IMPL(nmtkit::MLPAttention);

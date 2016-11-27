#include "config.h"

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

void MLPAttention::prepare(
    const vector<DE::Expression> & memories,
    dynet::ComputationGraph * cg) {
  // Concatenated memory matrix.
  // Shape: {memory_size, seq_length}
  i_concat_mem_ = DE::concatenate_cols(memories);

  // Dimention broadcasting matrix for the controller.
  // Shape: {1, seq_length}
  const unsigned seq_length = memories.size();
  i_broadcast_ = DE::input(
      *cg, {1, seq_length}, vector<float>(seq_length, 1.0f));

  // Parameters
  DE::Expression mem2h = DE::parameter(*cg, p_mem2h_);
  i_ctrl2h_ = DE::parameter(*cg, p_ctrl2h_);
  i_h2logit_ = DE::parameter(*cg, p_h2logit_);

  // Precomputes memory -> hidden mapping.
  // Shape: {hidden_size, seq_length}
  i_h_mem_ = mem2h * i_concat_mem_;
}

vector<DE::Expression> MLPAttention::compute(
    const DE::Expression & controller) {
  // Computes the attention distribution.
  // Shape: {hidden_size, 1}
  DE::Expression h_ctrl = i_ctrl2h_ * controller;
  // Shape: {hidden_size, seq_length}
  DE::Expression h = DE::tanh(DE::affine_transform(
        {i_h_mem_, h_ctrl, i_broadcast_}));
  // Shape: {seq_length, 1}
  DE::Expression atten_probs_inner = DE::softmax(DE::transpose(i_h2logit_ * h));

  // Computes the context vector.
  // Shape: {memory_size, 1}
  DE::Expression context_inner = i_concat_mem_ * atten_probs_inner;

  return {atten_probs_inner, context_inner};
}

}  // namespace nmtkit

NMTKIT_SERIALIZATION_IMPL(nmtkit::MLPAttention);

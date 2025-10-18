// =============================================
// cgadimpl/include/nn/nn.hpp
// =============================================
#pragma once
#include "tensor.hpp"
#include <utility>

namespace ag::nn {

// Elementwise activations
Tensor relu   (const Tensor& x);
Tensor leaky_relu(const Tensor& x, float alpha);
Tensor sigmoid(const Tensor& x);
Tensor tanh   (const Tensor& x);
Tensor softplus(const Tensor& x);
Tensor silu   (const Tensor& x);    // x * sigmoid(x)
Tensor gelu   (const Tensor& x);    // tanh approximation

// Rowwise reductions (B x C)
// Tensor row_max        (const Tensor& x);   // -> [B,1]
// Tensor row_sum        (const Tensor& x);   // -> [B,1]
Tensor logsumexp_row  (const Tensor& x);   // -> [B,1]
Tensor softmax_row    (const Tensor& x);   // -> [B,C]

// Loss (expects one-hot targets Y, same shape as logits Z)
Tensor cross_entropy_with_logits(const Tensor& logits, const Tensor& onehot_targets); // -> [1,1]

// Scaled Dot-Product Attention (single-head, 2D tensors)
// Q:[B,D], K:[C,D], V:[C,E], optional mask:[B,C] (1=keep, 0=mask), scale ~ 1/sqrt(D)
Tensor scaled_dot_product_attention(const Tensor& Q,
                                    const Tensor& K,
                                    const Tensor& V,
                                    const Tensor* mask,   // pass nullptr for no mask
                                    float scale);

} // namespace ag::nn

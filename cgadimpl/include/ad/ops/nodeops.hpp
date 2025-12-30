// =====================
// file: cgadimpl/include/ag/detail/nodeops.hpp
// =====================
#pragma once

#include "ad/core/graph.hpp"
#include "ad/autodiff/checkpoint.hpp"
#include "ad/utils/debug.hpp"

#include "ops/TensorOps.h"
#include "ops/UnaryOps/Arithmetics.h"
#include "ops/UnaryOps/Exponents.h"
#include "ops/UnaryOps/Reduction.h"
#include "ops/UnaryOps/Trigonometry.h"
#include "ops/Kernels.h"

#include <iostream>
#include <math.h>
#include <iterator>
#include <memory>

namespace ag {
namespace detail {

// Arithmetic Operations ------------------
// Binary ----------------
std::shared_ptr<Node> add_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b);
std::shared_ptr<Node> sub_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b);
std::shared_ptr<Node> mul_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b);
std::shared_ptr<Node> div_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b);

// inline functions --------------
inline std::shared_ptr<Node> operator+(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){ return add_nodeops(a,b);}
inline std::shared_ptr<Node> operator-(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){ return sub_nodeops(a,b);}
inline std::shared_ptr<Node> operator*(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){ return mul_nodeops(a,b);}
inline std::shared_ptr<Node> operator/(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){ return div_nodeops(a,b);}

// Unary Mathematical Functions -----------
std::shared_ptr<Node> exp_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> log_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> sqrt_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> reci_nodeops(const std::shared_ptr<Node>& a);
std::shared_ptr<Node> sign_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> abs_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> pow_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b);

// Linear algebra -------------
// Core Matrix Ops ------------
std::shared_ptr<Node> matmul_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b);
std::shared_ptr<Node> transpose_nodeops(const std::shared_ptr<Node>& x);

// Fused Ops --------------
std::shared_ptr<Node> linear_nodeops(const  std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c); // fused multiply-add a@b + c
std::shared_ptr<Node> fmab_nodeops(const  std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c); // fused multiply-add a@b + c

// activations ----------------------------
std::shared_ptr<Node> relu_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> tanh_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> sigmoid_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> softplus_nodeops(const std::shared_ptr<Node>& x);

// Modern Smooth Activations (better gradient flow) ---------
std::shared_ptr<Node> gelu_nodeops(const std::shared_ptr<Node>& x); // tanh approx
std::shared_ptr<Node> silu_nodeops(const std::shared_ptr<Node>& x); // x * sigmoid(x)
std::shared_ptr<Node> mish_nodeops(const std::shared_ptr<Node>& x);

// Parametric Activations ---
std::shared_ptr<Node> leaky_relu_nodeops(const std::shared_ptr<Node>& x, float alpha=0.01f); // alpha via const input

// Specialized Activations ---
std::shared_ptr<Node> gaus_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> parcon_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> lisht_nodeops(const std::shared_ptr<Node>& x);

// attention ---------
std::shared_ptr<Node> attention_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c, const std::shared_ptr<Node>& d);

// Gated Activations (transformer FFN blocks) ------------
std::shared_ptr<Node> swiglu_nodeops(const std::shared_ptr<Node>& x, const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c, const std::shared_ptr<Node>& d);

// Leaf -------------


// Loss ---------------
// Classification loss ----------------
std::shared_ptr<Node> cross_entropy_with_logits_nodeops(const std::shared_ptr<Node>& logits, const std::shared_ptr<Node>& onehot);
std::shared_ptr<Node> kldivergence_nodeops(const std::shared_ptr<Node>& logits,const std::shared_ptr<Node>& onehot);

// Regression loss --------------- 
std::shared_ptr<Node> mse_loss_nodeops( const std::shared_ptr<Node>& pred, const std::shared_ptr<Node>& target);
std::shared_ptr<Node> mae_loss_nodeops( const std::shared_ptr<Node>& pred, const std::shared_ptr<Node>& target);

// Layer norm --------------------
std::shared_ptr<Node> laynor_nodeops(const std::shared_ptr<Node>& x);

// RMS norm --------
std::shared_ptr<Node> rms_nodeops(const std::shared_ptr<Node>& x); // root mean square normalization
std::shared_ptr<Node> realrms_nodeops(const std::shared_ptr<Node>& x, float& g); // with learned scale

// Dynamic norm--------------
std::shared_ptr<Node> dyntanh_nodeops(const std::shared_ptr<Node>& x, float& a, float& b, float& g); // dynamic tanh via mean_all

// Reductions--------------
// Global reductions -------------
std::shared_ptr<Node> sum_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> mean_all_nodeops( const std::shared_ptr<Node>& x); // scalar

// Row-wise reductions ----------------
std::shared_ptr<Node> rowsum_nodeops(const std::shared_ptr<Node>& x); // [B,C] -> [B,1]
std::shared_ptr<Node> rowmax_nodeops(const std::shared_ptr<Node>& x); // [B,C] -> [B,1]

// Softmax (normalizing reductions) -----------------------
std::shared_ptr<Node> softmax_row_nodeops( const std::shared_ptr<Node>& z); // [B,C] -> [B,C]
std::shared_ptr<Node> logsumexp_row_nodeops(const std::shared_ptr<Node>& z); // [B,C] -> [B,1]

// trigonometric functions -------------
std::shared_ptr<Node> cos_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> sin_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> tan_nodeops(const std::shared_ptr<Node>& x);

// Hyperbolic functions -----------
std::shared_ptr<Node> sinh_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> cosh_nodeops(const std::shared_ptr<Node>& x);

// Inverse functions -------------
std::shared_ptr<Node> asin_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> acos_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> atan_nodeops(const std::shared_ptr<Node>& x);

// Inverse hyperbolic functions -------------
std::shared_ptr<Node> asinh_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> acosh_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> atanh_nodeops(const std::shared_ptr<Node>& x);

} // namespace detail
} // namespace ag
// =====================
// file: include/ag/detail/nodeops.hpp
// =====================
#pragma once

#include "ad/graph.hpp"
#include "ad/checkpoint.hpp"
#include "ad/kernels_api.hpp"
#include "ad/debug.hpp"
#include <iostream>
#include <math.h>
#include <iterator>
#include <memory>

namespace ag {
namespace detail {

    // --- Node-Level Operations (Internal API) ---
    // These functions perform the core graph manipulation and tensor computations.

//std::shared_ptr<Node> checkpoint_nodeops(const std::shared_ptr<Node> &v, const CheckpointOptions &opts);

std::shared_ptr<Node> add_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b);
std::shared_ptr<Node> sub_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b);
std::shared_ptr<Node> mul_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b);
std::shared_ptr<Node> div_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b);

std::shared_ptr<Node> relu_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> matmul_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b);
std::shared_ptr<Node> sum_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> flomul_nodeops(const std::shared_ptr<Node>& a, float b);
std::shared_ptr<Node> floadd_nodeops(float b, const std::shared_ptr<Node>& a);
std::shared_ptr<Node> flodiv_nodeops(float b, const std::shared_ptr<Node>& a);

inline std::shared_ptr<Node> operator+(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){ return add_nodeops(a,b);}
inline std::shared_ptr<Node> operator-(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){ return sub_nodeops(a,b);}
inline std::shared_ptr<Node> operator*(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){ return mul_nodeops(a,b);}
inline std::shared_ptr<Node> operator/(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){ return div_nodeops(a,b);}
inline std::shared_ptr<Node> operator*(const std::shared_ptr<Node>& a, float b){ return flomul_nodeops(a,b);}
inline std::shared_ptr<Node> operator*( float b, const std::shared_ptr<Node>& a){ return flomul_nodeops(a,b);}
inline std::shared_ptr<Node> operator/( float b, const std::shared_ptr<Node>& a){ return flodiv_nodeops(b,a);}
inline std::shared_ptr<Node> operator+( float b, const std::shared_ptr<Node>& a){ return floadd_nodeops(b,a);}
inline std::shared_ptr<Node> operator+( const std::shared_ptr<Node>& a, float b){ return floadd_nodeops(b,a);}

// unary elementwise
std::shared_ptr<Node> exp_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> log_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> tanh_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> gcu_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> mish_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> gaus_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> parcon_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> sigmoid_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> sign_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> softplus_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> relumask_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> cosh_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> sinh_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> cos_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> sin_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> sqrt_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> reci_nodeops(const std::shared_ptr<Node>& a);

std::shared_ptr<Node> linear_nodeops(const  std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c); // fused multiply-add a@b + c
std::shared_ptr<Node> moewe_nodeops(const std::shared_ptr<Node>& x, const std::shared_ptr<Node>& w, const std::shared_ptr<Node>& b);
std::shared_ptr<Node> reluatt_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c, const std::shared_ptr<Node>& d);
std::shared_ptr<Node> sigatt_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c, const std::shared_ptr<Node>& d);
std::shared_ptr<Node> gelu_nodeops(const std::shared_ptr<Node>& x); // tanh approx
std::shared_ptr<Node> silu_nodeops(const std::shared_ptr<Node>& x); // x * sigmoid(x)
std::shared_ptr<Node> leaky_relu_nodeops(const std::shared_ptr<Node>& x, float alpha=0.01f); // alpha via const input
std::shared_ptr<Node> lisht_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> transpose_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> swiglu_nodeops(const std::shared_ptr<Node>& x, const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c, const std::shared_ptr<Node>& d);
std::shared_ptr<Node> rms_nodeops(const std::shared_ptr<Node>& x); // root mean square normalization
std::shared_ptr<Node> realrms_nodeops(const std::shared_ptr<Node>& x, float& g); // with learned scale
std::shared_ptr<Node> dyntanh_nodeops(const std::shared_ptr<Node>& x, float& a, float& b, float& g); // dynamic tanh via mean_all
std::shared_ptr<Node> relaynor_nodeops(const std::shared_ptr<Node>& x, float& b, float& g); // with learned scale and bias
std::shared_ptr<Node> mambassm_nodeops(const std::shared_ptr<Node>& z, const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c, const std::shared_ptr<Node>& d); // state space model


// rowwise reductions / softmax family
std::shared_ptr<Node> rowsum_nodeops(const std::shared_ptr<Node>& x); // [B,C] -> [B,1]
std::shared_ptr<Node> rowmax_nodeops(const std::shared_ptr<Node>& x); // [B,C] -> [B,1]
std::shared_ptr<Node> mean_all_nodeops( const std::shared_ptr<Node>& x); // scalar
std::shared_ptr<Node> softmax_row_nodeops( const std::shared_ptr<Node>& z); // [B,C] -> [B,C]
std::shared_ptr<Node> logsumexp_row_nodeops(const std::shared_ptr<Node>& z); // [B,C] -> [B,1]
std::shared_ptr<Node> laynor_nodeops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> alibiatt_nodeops( const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c, const std::shared_ptr<Node>& d, float& m); // m = max seq len

// composite loss (one-hot targets)
std::shared_ptr<Node> cross_entropy_with_logits_nodeops(const std::shared_ptr<Node>& logits, const std::shared_ptr<Node>& onehot);
std::shared_ptr<Node> kldivergence_nodeops(const std::shared_ptr<Node>& logits,const std::shared_ptr<Node>& onehot);
std::shared_ptr<Node> fmab_nodeops(const  std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c); // fused multiply-add a@b + c
std::shared_ptr<Node> attention_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c, const std::shared_ptr<Node>& d);
std::shared_ptr<Node> mse_loss_nodeops( const std::shared_ptr<Node>& pred, const std::shared_ptr<Node>& target);
std::shared_ptr<Node> mae_loss_nodeops( const std::shared_ptr<Node>& pred, const std::shared_ptr<Node>& target);


} // namespace detail
} // namespace ag
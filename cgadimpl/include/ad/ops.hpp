// =====================
// file: include/ag/ops.hpp (declarations only)
// =====================
#pragma once
#include "ad/graph.hpp"
#include "ad/nodeops.hpp"
#include "ad/checkpoint.hpp"


namespace ag {

struct CheckpointOptions;

Value checkpoint(const Value &v, const CheckpointOptions &opts);

Value add (const Value& a, const Value& b);
Value sub (const Value& a, const Value& b);
Value mul (const Value& a, const Value& b);
Value relu (const Value& x);
Value matmul(const Value& a, const Value& b);
Value sum (const Value& x);
Value flomul (const Value& a, float b);

inline Value operator+(const Value& a, const Value& b){ return add(a,b);}
inline Value operator-(const Value& a, const Value& b){ return sub(a,b);}
inline Value operator*(const Value& a, const Value& b){ return mul(a,b);}
inline Value operator*(const Value& a, float b){ return flomul(a,b);}

// unary elementwise
Value exp (const Value& x);
Value log (const Value& x);
Value tanh (const Value& x);
Value gcu (const Value& x);
Value mish (const Value& x);
Value gaus (const Value& x);
Value parcon(const Value& x);
Value sigmoid(const Value& x);
Value softplus(const Value& x);
Value gelu (const Value& x); // tanh approx
Value silu (const Value& x); // x * sigmoid(x)
Value leaky_relu(const Value& x, float alpha=0.01f); // alpha via const input
Value lisht(const Value& x);
Value transpose(const Value& x);
Value swiglu(const Value& x, const Value& a, const Value& b, const Value& c, const Value& d);
Value deconval(const Value& x, float g); // alpha via const input
Value rms(const Value& x); // root mean square normalization
Value realrms(const Value& x, float g); // with learned scale
Value dyntanh(const Value& x, float a, float b, float g); // dynamic tanh via mean_all
Value relaynor(const Value& x, float b, float g); // with learned scale and bias
Value mambassm(const Value& z, const Value& a, const Value& b, const Value& c, const Value& d); // state space model


// rowwise reductions / softmax family
Value rowsum (const Value& x); // [B,C] -> [B,1]
Value rowmax (const Value& x); // [B,C] -> [B,1]
Value mean_all(const Value& x); // scalar
Value softmax_row(const Value& z); // [B,C] -> [B,C]
Value logsumexp_row(const Value& z); // [B,C] -> [B,1]
Value laynor(const Value& x);
Value alibiatt(const Value& a, const Value& b, const Value& c, const Value& d, float m); // m = max seq len

// composite loss (one-hot targets)
Value cross_entropy_with_logits(const Value& logits, const Value& onehot);
Value kldivergence(const Value& logits, const Value& onehot);
Value fmab(const Value& a, const Value& b, const Value& c); // fused multiply-add a@b + c
Value attention(const Value& a, const Value& b, const Value& c, const Value& d);
Value mse_loss(const Value& pred, const Value& target);
Value mae_loss(const Value& pred, const Value& target);


Tensor forward_eval_node(Node* node);


} // namespace ag
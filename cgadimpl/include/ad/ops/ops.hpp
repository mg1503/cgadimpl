// =====================
// file: cgadimpl/include/ag/ops.hpp (declarations only)
// =====================
#pragma once
#include "ad/core/graph.hpp"
#include "ad/ops/nodeops.hpp"
#include "ad/autodiff/checkpoint.hpp"


namespace ag {

struct CheckpointOptions;

Value checkpoint(const Value &v, const CheckpointOptions &opts);
Value inplace_checkpoint(const Value& v);
Value add (const Value& a, const Value& b);
Value sub (const Value& a, const Value& b);
Value mul (const Value& a, const Value& b);
Value div (const Value& a, const Value& b);

Value relu (const Value& x);
Value matmul(const Value& a, const Value& b);
Value sum (const Value& x);
Value abs (const Value& x);

// Binary operators now rely on implicit conversion from float to Value
inline Value operator+(const Value& a, const Value& b){ return ag::add(a,b);}
inline Value operator-(const Value& a, const Value& b){ return ag::sub(a,b);}
inline Value operator*(const Value& a, const Value& b){ return ag::mul(a,b);}
inline Value operator/(const Value& a, const Value& b){ return ag::div(a,b);}

// unary elementwise
Value exp (const Value& x);
Value log (const Value& x);
Value tanh (const Value& x);
Value mish (const Value& x);
Value gaus (const Value& x);
Value parcon(const Value& x);
Value sigmoid(const Value& x);

Value sin(const Value& x);
Value tan(const Value& x);
Value asin(const Value& x);
Value acos(const Value& x);
Value atan(const Value& x);
Value asinh(const Value& x);
Value acosh(const Value& x);
Value cosh(const Value& x);
Value sinh(const Value& x);

Value softplus(const Value& x);

Value gelu (const Value& x); // tanh approx
Value silu (const Value& x); // x * sigmoid(x)
Value leaky_relu(const Value& x, float alpha=0.01f); // alpha via const input
Value lisht(const Value& x);
Value transpose(const Value& x);
Value swiglu(const Value& x, const Value& a, const Value& b, const Value& c, const Value& d);
Value rms(const Value& x); // root mean square normalization
Value realrms(const Value& x, float g); // with learned scale
Value dyntanh(const Value& x, float a, float b, float g); // dynamic tanh via mean_all
Value sign (const Value& x);


// rowwise reductions / softmax family
Value rowsum (const Value& x); // [B,C] -> [B,1]
Value rowmax (const Value& x); // [B,C] -> [B,1]
Value mean_all(const Value& x); // scalar
Value softmax_row(const Value& z); // [B,C] -> [B,C]
Value logsumexp_row(const Value& z); // [B,C] -> [B,1]
Value laynor(const Value& x);


// composite loss (one-hot targets)
Value cross_entropy_with_logits(const Value& logits, const Value& onehot);
Value kldivergence(const Value& logits, const Value& onehot);
Value fmab(const Value& a, const Value& b, const Value& c); // fused multiply-add a@b + c
Value linear(const Value& a, const Value& b, const Value& c); // fused multiply-add a@b + c

Value attention(const Value& a, const Value& b, const Value& c, const Value& d);
Value mse_loss(const Value& pred, const Value& target);
Value mae_loss(const Value& pred, const Value& target);


Tensor forward_eval_node(Node* node);


} // namespace ag
// =====================
// file: include/ag/ops.hpp (declarations only)
// =====================
#pragma once
#include "ad/graph.hpp"


namespace ag {


Value add (const Value& a, const Value& b);
Value sub (const Value& a, const Value& b);
Value mul (const Value& a, const Value& b);
Value relu (const Value& x);
Value matmul(const Value& a, const Value& b);
Value sum (const Value& x);


inline Value operator+(const Value& a, const Value& b){ return add(a,b);}
inline Value operator-(const Value& a, const Value& b){ return sub(a,b);}
inline Value operator*(const Value& a, const Value& b){ return mul(a,b);}

// unary elementwise
Value exp (const Value& x);
Value log (const Value& x);
Value tanh (const Value& x);
Value gcu (const Value& x);
Value sigmoid(const Value& x);
Value softplus(const Value& x);
Value gelu (const Value& x); // tanh approx
Value silu (const Value& x); // x * sigmoid(x)
Value leaky_relu(const Value& x, float alpha=0.01f); // alpha via const input


// rowwise reductions / softmax family
Value rowsum (const Value& x); // [B,C] -> [B,1]
Value rowmax (const Value& x); // [B,C] -> [B,1]
Value mean_all(const Value& x); // scalar
Value softmax_row(const Value& z); // [B,C] -> [B,C]
Value logsumexp_row(const Value& z); // [B,C] -> [B,1]


// composite loss (one-hot targets)
Value cross_entropy_with_logits(const Value& logits, const Value& onehot);
Value kldivergence(const Value& logits, const Value& onehot);
Value fmab(const Value& a, const Value& b, const Value& c); // fused multiply-add a@b + c
Value attention(const Value& a, const Value& b, const Value& c, const Value& d);
Value mse_loss(const Value& pred, const Value& target);
Value mae_loss(const Value& pred, const Value& target);

} // namespace ag
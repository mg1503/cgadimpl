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

// Binary arith -----------------
Value add (const Value& a, const Value& b);
Value sub (const Value& a, const Value& b);
Value mul (const Value& a, const Value& b);
Value div (const Value& a, const Value& b);


// Classic Activations ---------------
Value relu (const Value& x);
Value sigmoid(const Value& x);
Value tanh (const Value& x);
Value softplus(const Value& x);

// Smooth Activations (better gradient flow) ---
Value gelu (const Value& x); // tanh approx
Value silu (const Value& x); // x * sigmoid(x)
Value mish (const Value& x);

// Parametric Activations ------
Value leaky_relu(const Value& x, float alpha=0.01f); // alpha via const input

// Specialized activations -----------
Value gaus (const Value& x);
Value parcon(const Value& x);
Value lisht(const Value& x);

// Standard Attention ---------
Value attention(const Value& a, const Value& b, const Value& c, const Value& d);

// Gated activation -----------------
Value swiglu(const Value& x, const Value& a, const Value& b, const Value& c, const Value& d);

//Leaf --------------------
//leaf

//Unary Mathematical Functions ------------------
Value exp (const Value& x);
Value log (const Value& x);
Value sqrt(const Value& x);
Value reci(const Value& x);
Value sign (const Value& x);
Value abs(const Value& x);
Value pow(const Value& a, const Value& b);

//Core Matrix Operations ------------------------
Value matmul(const Value& a, const Value& b);
Value transpose(const Value& x);

//Fused Operations (better performance, fewer memory accesses) ---------------
Value linear(const Value& a, const Value& b, const Value& c);
Value fmab(const Value& a, const Value& b, const Value& c); // fused multiply-add a@b + c

//Classification losses ---------------
Value cross_entropy_with_logits(const Value& logits, const Value& onehot);
Value kldivergence(const Value& logits, const Value& onehot);

//Regression Losses --------------
Value mse_loss(const Value& pred, const Value& target);
Value mae_loss(const Value& pred, const Value& target);

//Layer Normalization ------------
Value laynor(const Value& x); 

//RMS Normalization -------------
Value rms(const Value& x); // root mean square normalization
Value realrms(const Value& x, float g); // with learned scale

//Dynamic Normalization --------------
Value dyntanh(const Value& x, float a, float b, float g); // dynamic tanh via mean_all

//Global Reductions -------------------
Value sum(const Value& x);
Value mean_all(const Value& x); // scalar

//Row-wise Reductions ------------------------
Value rowsum (const Value& x); // [B,C] -> [B,1]
Value rowmax (const Value& x); // [B,C] -> [B,1] 

//Softmax Family ---------------
Value softmax_row(const Value& z); // [B,C] -> [B,C]
Value logsumexp_row(const Value& z); // [B,C] -> [B,1]

//Trigonometric Functions --------------------
Value sin(const Value& x);
Value cos(const Value& x);
Value tan(const Value& x);

//Hyperbolic Functions ------------------
Value cosh(const Value& x);
Value sinh(const Value& x); 

//Inverse Trigonometric Functions --------------
Value asin(const Value& x);
Value acos(const Value& x);
Value atan(const Value& x);

//Inverese Hyperbolic Trigonometric Functions ----------------
Value asinh(const Value& x);
Value acosh(const Value& x); 
Value atanh(const Value& x);




// Binary operators now rely on implicit conversion from float to Value
inline Value operator+(const Value& a, const Value& b){ return ag::add(a,b);}
inline Value operator-(const Value& a, const Value& b){ return ag::sub(a,b);}
inline Value operator*(const Value& a, const Value& b){ return ag::mul(a,b);}
inline Value operator/(const Value& a, const Value& b){ return ag::div(a,b);}

Tensor forward_eval_node(Node* node);


} // namespace ag
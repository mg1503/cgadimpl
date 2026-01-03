// ====================================================================
// FILE: cgadimpl/src/autodiff/autodiff_jvp_ops.cpp (GPU-Aware Version)
// ====================================================================
#include "ad/detail/autodiff_ops.hpp"
#include <stdexcept> // Required for std::runtime_error

#include "ad/runtime/cuda_graphs.hpp"

namespace ag {
namespace detail{

// The 'T' shorthand is still useful, so we keep it.
inline const Tensor& T(const std::function<const Tensor&(Node*)>& f, Node* p){ return f(p); }

//Binary arith ----
Tensor jvp_Add(Node* n, const std::function<const Tensor&(Node*)>& t){
    // The '+' operator now handles both CPU and GPU, and is stream-aware.
    return T(t, n->inputs[0].get()) + T(t, n->inputs[1].get());
}
Tensor jvp_Sub(Node* n, const std::function<const Tensor&(Node*)>& t){
    // The '-' operator now handles both CPU and GPU.
    return T(t, n->inputs[0].get()) - T(t, n->inputs[1].get());
}
Tensor jvp_Mul(Node* n, const std::function<const Tensor&(Node*)>& t){ 
    Node* A = n->inputs[0].get();
    Node* B = n->inputs[1].get();

    // The '*' and '+' operators now handle both CPU and GPU.
    // The entire expression works on either device and is fully asynchronous on CUDA.
    return (T(t, A) * B->value) + (A->value * T(t, B));
}
Tensor jvp_Div(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* A_node = n->inputs[0].get();
    Node* B_node = n->inputs[1].get();
    const Tensor& A = A_node->value;
    const Tensor& B = B_node->value;
    
    return (T(t, A_node) / B) - (A * T(t, B_node) / (B * B));
}

//classic activations--------
Tensor jvp_Relu(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X = n->inputs[0].get();
    cudaStream_t stream = (cudaStream_t)ag::current_stream();
    // Create the ReLU mask using pure arithmetic
    const float epsilon = 1e-9f;
    Tensor sign_X = X->value / (OwnTensor::abs(X->value, stream) + epsilon);
    Tensor mask = (sign_X + OwnTensor::abs(sign_X, stream)) * 0.5f; // relu(sign(x))

    return T(t, X) * mask;
}

Tensor jvp_Sigmoid(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X = n->inputs[0].get();
    const Tensor& s = n->value; // s is the result of sigmoid(x) from the forward pass
    
    // The operators handle broadcasting and device dispatch automatically.
    return T(t, X) * (s * (1.0f - s));
}

Tensor jvp_Tanh(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X = n->inputs[0].get();
    const Tensor& th = n->value;
    
    // The operators handle broadcasting and device dispatch automatically.
    return T(t, X) * (1.0f - (th * th));
}


Tensor jvp_Softplus(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X = n->inputs[0].get();

    // Re-implement sigmoid using OwnTensor ops
    Tensor sig_x = 1.0f / (1.0f + OwnTensor::exp(X->value * -1.0f));
    
    return T(t, X) * sig_x;
}

// Smooth Activations -------
Tensor jvp_GELU(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X_node = n->inputs[0].get();
    const Tensor& x = X_node->value;

    const float c1 = 0.7978845608f; // sqrt(2.0f / M_PI)
    const float c2 = 0.044715f;

    // Recompute intermediates
    Tensor x2 = x * x;
    Tensor x3 = x2 * x;
    Tensor u = (x + x3 * c2) * c1;
    Tensor th_u = OwnTensor::tanh(u);
    
    // Compute du/dx = c1 * (1 + 3 * c2 * x^2)
    Tensor du_dx = (1.0f + (x2 * (3.0f * c2))) * c1;

    // Compute the full GELU derivative
    Tensor d_gelu = (1.0f + th_u) * 0.5f + (x * (1.0f - (th_u * th_u)) * du_dx) * 0.5f;

    return T(t, X_node) * d_gelu;
}

Tensor jvp_SiLU(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X = n->inputs[0].get();
    const Tensor& x_val = X->value;

    // Re-implement sigmoid and its derivative
    Tensor s = 1.0f / (1.0f + OwnTensor::exp(x_val * -1.0f));
    Tensor s_prime = s * (1.0f - s);
    
    // Full SiLU derivative
    Tensor d_silu = s + x_val * s_prime;
    
    return T(t, X) * d_silu;
}
Tensor jvp_Mish(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X_node = n->inputs[0].get();
    const Tensor& x = X_node->value;

    // Re-calculate intermediates needed for the derivative
    Tensor sp = OwnTensor::log(1.0f + OwnTensor::exp(x));
    Tensor tanh_sp = OwnTensor::tanh(sp);
    Tensor sig_x = 1.0f / (1.0f + OwnTensor::exp(x * -1.0f));

    // The derivative of mish
    Tensor d_mish = tanh_sp + x * (1.0f - (tanh_sp * tanh_sp)) * sig_x;
    
    return T(t, X_node) * d_mish;
}

// Parametric Activations ------

Tensor jvp_LeakyRelu(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X_node = n->inputs[0].get();
    Node* A_node = n->inputs[1].get();
    const Tensor& x = X_node->value;
    float alpha = A_node->value.to_cpu().data<float>()[0];

    cudaStream_t stream = (cudaStream_t)ag::current_stream();

    // --- Create the Leaky ReLU derivative mask using pure arithmetic ---
    const float epsilon = 1e-9f;
    Tensor sign_x = x / (OwnTensor::abs(x, stream) + epsilon);
    Tensor mask_pos = (sign_x + OwnTensor::abs(sign_x, stream)) * 0.5f;
    Tensor mask_neg = 1.0f - mask_pos;
    Tensor d_leaky = mask_pos + (mask_neg * alpha);
    
    return T(t, X_node) * d_leaky;
}

// Specialized activations ----------

Tensor jvp_Gaus(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X = n->inputs[0].get();
    
    // JVP is tangent(x) * (-2 * x * exp(-x^2))
    // n->value from the forward pass already holds exp(-x^2).
    return T(t, X) * -2.0f * X->value * n->value;
}

Tensor jvp_Parcon(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X = n->inputs[0].get();
    return T(t, X) * (2.0f - 2.0f * X->value);
}

Tensor jvp_LiSHT(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X = n->inputs[0].get();
    const Tensor& x_val = X->value;

    Tensor th_x = OwnTensor::tanh(x_val);
    Tensor ch_x = OwnTensor::cosh(x_val);
    Tensor sech_x_sq = 1.0f / (ch_x * ch_x);
    
    Tensor d_lisht = th_x + x_val * sech_x_sq;
    
    return T(t, X) * d_lisht;
}

// Standard Attention ---------

Tensor jvp_Attention(Node* n, const std::function<const Tensor&(Node*)>& t){
    throw std::runtime_error("JVP for Attention not implemented yet!");
}

// Gated activation ------------
Tensor jvp_SWIGLU(Node* n, const std::function<const Tensor&(Node*)>& t){
    throw std::runtime_error("JVP for SWIGLU not implemented yet!");
}

//Leaf --------
Tensor jvp_Leaf(Node*, const std::function<const Tensor&(Node*)>&){
    return Tensor(Shape{}, TensorOptions{}); // unused
}
 
//Unary Mathematical Functions ------------------
Tensor jvp_Exp(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X = n->inputs[0].get();
    // The '*' operator is device-agnostic and stream-aware.
    return T(t, X) * n->value;
}

Tensor jvp_Log(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X = n->inputs[0].get();
    // The '/' operator is device-agnostic and stream-aware.
    return T(t, X) / X->value;
}

Tensor jvp_Sqrt(Node* n, const std::function<const Tensor&(Node*)>& t){
    // JVP is tangent(x) * (0.5 / sqrt(x)) = tangent(x) * 0.5 / y
    return T(t, n->inputs[0].get()) * 0.5f / n->value;
}
Tensor jvp_Reciprocal(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X_node = n->inputs[0].get();
    const Tensor& X = X_node->value;
    
    return (T(t, X_node) * -1.0f) / (X * X);
}
Tensor jvp_Sign(Node* n, const std::function<const Tensor&(Node*)>& t){
    // The JVP is 0. We create a zero-filled tensor with the correct shape and device.
    return OwnTensor::Tensor::zeros(n->value.shape(), ag::options(n->value));
}
Tensor jvp_Abs(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X = n -> inputs[0].get();

    const float epsilon = 1e-9f;
    Tensor sign_x = X -> value / (n->value + epsilon);

    return T(t, X)* sign_x;
}

Tensor jvp_Pow(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* A_node = n->inputs[0].get(); // base a
    Node* B_node = n->inputs[1].get(); // exponent b
    const Tensor& a = A_node->value;
    const Tensor& b = B_node->value;
    const Tensor& y = n->value; // y = a^b
    
    // JVP = b * (y / a) * tangent(a) + y * log(a) * tangent(b)
    Tensor term1 = b * (y / a) * T(t, A_node);
    Tensor term2 = y * OwnTensor::log(a, ag::current_stream()) * T(t, B_node);
    
    return term1 + term2;
}

//Core Matrix Operations ------------------------
Tensor jvp_MatMul(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* A = n->inputs[0].get();
    Node* B = n->inputs[1].get();
    return OwnTensor::matmul(T(t, A), B->value) + OwnTensor::matmul(A->value, T(t, B));
}

Tensor jvp_Transpose(Node* n, const std::function<const Tensor&(Node*)>& t){
    // Call the .t() member function on the tangent tensor.
    return T(t, n->inputs[0].get()).t();
}
//Fused Operations (better performance, fewer memory accesses) -------------
Tensor jvp_Linear(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* A = n->inputs[0].get();
    Node* B = n->inputs[1].get();
    Node* C = n->inputs[2].get();
    
    Tensor dA = OwnTensor::matmul(T(t, A), B->value);
    Tensor dB = OwnTensor::matmul(A->value, T(t, B));
    
    return dA + dB + T(t, C);
}

Tensor jvp_FMA(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* A = n->inputs[0].get();
    Node* B = n->inputs[1].get();
    Node* C = n->inputs[2].get();
    
    Tensor dA = OwnTensor::matmul(T(t, A), B->value.t());
    Tensor dB = OwnTensor::matmul(A->value, T(t, B).t());
    
    return dA + dB + T(t, C);
}
//Classification losses --------------- 

Tensor jvp_CeWithLogits(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* Z_node = n->inputs[0].get();
    Node* Y_node = n->inputs[1].get();
    const Tensor& Z = Z_node->value;
    const Tensor& Y = Y_node->value;
    
    const float inv_batch_size = 1.0f / static_cast<float>(Z.shape().dims[0]);

    // --- VJP for Z ---
    // Re-calculate stable softmax
    Tensor max_val_z = OwnTensor::reduce_max(Z, {-1}, true);
    Tensor exp_z = OwnTensor::exp(Z - max_val_z);
    Tensor softmax_z = exp_z / OwnTensor::reduce_sum(exp_z, {-1}, true);
    Tensor gZ = (softmax_z - Y) * inv_batch_size;
    
    // --- VJP for Y ---
    // Re-calculate stable log_softmax
    Tensor log_sum_exp_z = OwnTensor::log(OwnTensor::reduce_sum(exp_z, {-1}, true)) + max_val_z;
    Tensor log_softmax_z = Z - log_sum_exp_z;
    Tensor gY = log_softmax_z * (-1.0f * inv_batch_size);

    // --- Dot product with tangents ---
    Tensor dot_Z = OwnTensor::reduce_sum(gZ * t(Z_node));
    Tensor dot_Y = OwnTensor::reduce_sum(gY * t(Y_node));

    return dot_Z + dot_Y;
}
Tensor jvp_KLDivergence(Node* n, const std::function<const Tensor&(Node*)>& t){
    throw std::runtime_error("JVP for KLDivergence not implemented yet!");
}
//Regression Losses --------------
Tensor jvp_MSELoss(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* Z_node = n->inputs[0].get(); // Predictions
    Node* Y_node = n->inputs[1].get(); // Targets
    const Tensor& Z = Z_node->value;
    const Tensor& Y = Y_node->value;
    
    // N is the total number of elements for averaging.
    const float N = static_cast<float>(Z.numel());
    Tensor diff = Z - Y;
    
    // Calculate the partial derivatives (gradients) of the loss w.r.t. Z and Y.
    // dL/dZ = (2/N) * (Z - Y)
    Tensor gZ = diff * (2.0f / N);
    // dL/dY = (-2/N) * (Z - Y)
    Tensor gY = diff * (-2.0f * N);

    // The JVP is the sum of dot products: (dL/dZ ⋅ tZ) + (dL/dY ⋅ tY)
    // A dot product is an element-wise multiplication followed by a sum.
    Tensor dot_Z = OwnTensor::reduce_sum(gZ * t(Z_node));
    Tensor dot_Y = OwnTensor::reduce_sum(gY * t(Y_node));

    // The result is a scalar tensor.
    return dot_Z + dot_Y;
}
Tensor jvp_MAELoss(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* Z_node = n->inputs[0].get();
    Node* Y_node = n->inputs[1].get();
    const Tensor& Z = Z_node->value;
    const Tensor& Y = Y_node->value;
    
    const float inv_N = 1.0f / static_cast<float>(Z.numel());

    // Arithmetically implemented sign function
    const float epsilon = 1e-9f;
    Tensor diff = Z - Y;
    
    cudaStream_t stream = (cudaStream_t)ag::current_stream();
    Tensor sign_diff = diff / (OwnTensor::abs(diff, stream) + epsilon);
    
    // VJP parts
    Tensor gZ = sign_diff * inv_N;
    Tensor gY = sign_diff * (-1.0f * inv_N);

    // Dot product with tangents
    Tensor dot_Z = OwnTensor::reduce_sum(gZ * t(Z_node));
    Tensor dot_Y = OwnTensor::reduce_sum(gY * t(Y_node));
    
    return dot_Z + dot_Y;
}
//Layer Normalization ------------

Tensor jvp_LayerNorm(Node* n, const std::function<const Tensor&(Node*)>& t){
    throw std::runtime_error("JVP for LayerNorm not implemented yet!");
}

//RMS Normalization -------------
Tensor jvp_RMSNorm(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X_node = n->inputs[0].get();
    const Tensor& x = X_node->value;

    const Tensor& rms = *n->tape[0];
    const Tensor& y   = *n->tape[1];
    
    // 'dot' is the row-wise sum of (tangent(x) * y)
    Tensor dot = OwnTensor::reduce_sum(T(t, X_node) * y, {-1}, true);
    
    const float N = static_cast<float>(x.shape().dims.back());

    return (T(t, X_node) / rms) - (y * dot / (rms * N));
}
Tensor jvp_RealRMSNorm(Node* n, const std::function<const Tensor&(Node*)>& t){
    throw std::runtime_error("JVP for RealRMSNorm not implemented yet!");
}

//Dynamic Normalization --------------
Tensor jvp_Dyntanh(Node* n, const std::function<const Tensor&(Node*)>& t){
    throw std::runtime_error("JVP for Dyntanh not implemented yet!");
}   

//Global Reductions -------------------
Tensor jvp_Sum(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X = n->inputs[0].get();
    return OwnTensor::reduce_sum(t(X));
}
Tensor jvp_MeanAll(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X = n->inputs[0].get();
    return OwnTensor::reduce_mean(t(X));
}

//Row-wise Reductions ------------------------
Tensor jvp_RowSum(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X = n->inputs[0].get();
    return OwnTensor::reduce_sum(t(X), {-1}, true); // Sum over the last dimension and keep it
}
Tensor jvp_RowMax(Node* n, const std::function<const Tensor&(Node*)>& t){
    // To implement this, we need to know the *index* of the max element in each row.
    // This requires an `argmax` function or comparison operators, which are not 
    // available in OwnTensor.
    throw std::runtime_error("JVP for RowMax cannot be implemented without argmax or comparison ops in the tensor library.");
}
//Softmax Family ---------------
Tensor jvp_SoftmaxRow(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* Z_node = n->inputs[0].get();
    const Tensor& y = n->value; // y = softmax(z) from forward pass
    const Tensor& tZ = t(Z_node); // tangent(z)

    // dot product along the rows
    Tensor dot = OwnTensor::reduce_sum(y * tZ, {-1}, true);
    
    return y * (tZ - dot);
}
Tensor jvp_LogSumExpRow(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* Z_node = n->inputs[0].get();
    const Tensor& Z = Z_node->value;
    const Tensor& tZ = t(Z_node); // tangent(z)
    
    // Recompute stable softmax(z)
    Tensor max_val = OwnTensor::reduce_max(Z, {-1}, true);
    Tensor exp_z = OwnTensor::exp(Z - max_val);
    Tensor sum_exp_z = OwnTensor::reduce_sum(exp_z, {-1}, true);
    Tensor y = exp_z / sum_exp_z;

    // Return the row-wise dot product
    return OwnTensor::reduce_sum(y * tZ, {-1}, true);
}
//Trigonometric Functions --------------------
Tensor jvp_Sin(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X = n->inputs[0].get();
    return T(t, X) * OwnTensor::cos(X->value);
}
Tensor jvp_Cos(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X = n->inputs[0].get();
    return T(t, X) * -1.0f * OwnTensor::sin(X->value);
}
Tensor jvp_Tan(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X = n->inputs[0].get();
    Tensor cos_x = OwnTensor::cos(X->value);
    // JVP is tangent(x) * (1/cos(x)^2)
    return T(t, X) * (1.0f / (cos_x * cos_x));
}

//Hyperbolic Functions ------------------
Tensor jvp_Cosh(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X = n->inputs[0].get();
    return T(t, X) * OwnTensor::sinh(X->value);
}
Tensor jvp_Sinh(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X = n->inputs[0].get();
    return T(t, X) * OwnTensor::cosh(X->value);
}
//Inverse Trigonometric Functions --------------
Tensor jvp_Asin(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X = n->inputs[0].get();
    // JVP is tangent(x) * (1 / sqrt(1 - x^2))
    return T(t, X) * (1.0f / OwnTensor::sqrt(1.0f - (X->value * X->value), ag::current_stream()));
}

Tensor jvp_Acos(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X = n->inputs[0].get();
    // JVP is tangent(x) * (-1 / sqrt(1 - x^2))
    return T(t, X) * (-1.0f / OwnTensor::sqrt(1.0f - (X->value * X->value), ag::current_stream()));
}

Tensor jvp_Atan(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X = n->inputs[0].get();
    // JVP is tangent(x) * (1 / (1 + x^2))
    return T(t, X) * (1.0f / (1.0f + (X->value * X->value)));
}
//Inverse Hyperbolic Trigonometric Functions ----------------
Tensor jvp_ASinh(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X = n->inputs[0].get();
    // JVP is tangent(x) * (1 / (1 + x^2))
    return T(t, X) / OwnTensor::sqrt((X->value * X-> value) + 1.0f, ag::current_stream());
}
Tensor jvp_ACosh(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X = n->inputs[0].get();
    // JVP is tangent(x) * (1 / (1 + x^2))
    return T(t, X) / OwnTensor::sqrt((X->value * X-> value) - 1.0f, ag::current_stream());
}
Tensor jvp_ATanh(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X = n->inputs[0].get();
    // JVP = tangent(x) * (1 / (1 - x^2))
    return T(t, X) / (1.0f - (X->value * X->value));
}



} // namespace detail


// -------- dispatch table --------
JvpFn jvp_lookup(Op op){
    switch(op){
#define OP(name, arity, str) case Op::name: return &detail::jvp_##name;
#include "ad/detail/ops.def"
#undef OP
        default: return nullptr;
    }
}

} // namespace ag
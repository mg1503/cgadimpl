// ====================================================================
// FILE: cgadimpl/src/autodiff/autodiff_vjp_ops.cpp (GPU-Aware Version)
// ====================================================================

#include "ad/detail/autodiff_ops.hpp"
#include "ad/runtime/cuda_graphs.hpp"
#include <cmath>
#include <stdexcept> 

namespace ag {
namespace detail{

static Tensor reduce_for_broadcast(const Tensor& grad_in, const Tensor& target_val) {
    if (grad_in.shape().dims == target_val.shape().dims) {
        return grad_in;
    }
    const auto& grad_dims = grad_in.shape().dims;
    const auto& target_dims = target_val.shape().dims;
    std::vector<int64_t> axes_to_sum;
    int grad_ndim = grad_dims.size();
    int target_ndim = target_dims.size();
    for (int i = 0; i < grad_ndim; ++i) {
        int target_dim_idx = i + target_ndim - grad_ndim;
        if (target_dim_idx < 0 || (target_dims[target_dim_idx] == 1 && grad_dims[i] > 1)) {
            axes_to_sum.push_back(i);
        }
    }
    Tensor summed_grad = grad_in;
    if (!axes_to_sum.empty()) {
        summed_grad = OwnTensor::reduce_sum(grad_in, axes_to_sum, true);
    }
    return summed_grad.reshape(target_val.shape());
}
// --- Basic Arithmetic ---
void vjp_Add(Node* n, const Tensor& gy){
    Node* A = n->inputs[0].get(); 
    Node* B = n->inputs[1].get();
    if (A->requires_grad()) A->grad += reduce_for_broadcast(gy, A->value);
    if (B->requires_grad()) B->grad += reduce_for_broadcast(gy, B->value);
}

void vjp_Sub(Node* n, const Tensor& gy){
    Node* A = n->inputs[0].get(); 
    Node* B = n->inputs[1].get();
    if (A->requires_grad()) A->grad += reduce_for_broadcast(gy, A->value);
    if (B->requires_grad()) B->grad -= reduce_for_broadcast(gy, B->value);
}

void vjp_Mul(Node* n, const Tensor& gy){
    Node* A = n->inputs[0].get(); 
    Node* B = n->inputs[1].get();
    if (A->requires_grad()) A->grad += reduce_for_broadcast(gy * B->value, A->value);
    if (B->requires_grad()) B->grad += reduce_for_broadcast(gy * A->value, B->value);
}

void vjp_Div(Node* n, const Tensor& gy){
    Node* A_node = n->inputs[0].get();
    Node* B_node = n->inputs[1].get();
    const Tensor& A = A_node->value;
    const Tensor& B = B_node->value;
    if (A_node->requires_grad()) {
        A_node->grad += reduce_for_broadcast(gy / B, A);
    }
    if (B_node->requires_grad()) {
        B_node->grad -= reduce_for_broadcast(gy * A / (B * B), B);
    }
}

// Classic Activations ---------------
void vjp_Relu(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (X->requires_grad()) {
        Tensor mask = OwnTensor::Tensor::zeros(X->value.shape(), ag::options(X->value));
        if (X->value.is_cpu()) {
            dispatch_by_dtype(X->value.dtype(), [&](auto dummy){
                using T = decltype(dummy);
                const T* x_ptr = X->value.data<T>();
                T* m_ptr = mask.data<T>();
                for(int64_t i=0; i<X->value.numel(); ++i) {
                    if constexpr (std::is_same_v<T, OwnTensor::complex32_t> || 
                                  std::is_same_v<T, OwnTensor::complex64_t> || 
                                  std::is_same_v<T, OwnTensor::complex128_t>) {
                        if (x_ptr[i].real() > 0) m_ptr[i] = T(1.0f);
                    } else {
                        if (x_ptr[i] > T(0)) m_ptr[i] = T(1.0f);
                    }
                }
            });
        }
        X->grad += gy * mask;
    }
}

void vjp_Sigmoid(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (X->requires_grad()) {
        const Tensor& y = n->value;
        X->grad += gy * y * (1.0f - y);
    }
}
void vjp_Tanh(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (X->requires_grad()) X->grad += gy * (1.0f - n->value * n->value);
}
void vjp_Softplus(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (X->requires_grad()) {
        Tensor sig = 1.0f / (1.0f + OwnTensor::exp(X->value * -1.0f, ag::current_stream()));
        X->grad += gy * sig;
    }
}

// Smooth Activations (better gradient flow) ---
void vjp_GELU(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (X->requires_grad()) {
        const float c1 = 0.7978845608f; 
        const float c2 = 0.044715f;
        Tensor x = X->value;
        Tensor x3 = x * x * x;
        Tensor u = c1 * (x + c2 * x3);
        Tensor tu = OwnTensor::tanh(u);
        Tensor d_tu = 1.0f - tu * tu;
        Tensor du_dx = c1 * (1.0f + 3.0f * c2 * x * x);
        X->grad += gy * (0.5f * (1.0f + tu) + 0.5f * x * d_tu * du_dx);
    }
}

void vjp_SiLU(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (X->requires_grad()) {
        Tensor x = X->value;
        Tensor sig = 1.0f / (1.0f + OwnTensor::exp(x * -1.0f, ag::current_stream()));
        X->grad += gy * (sig + x * sig * (1.0f - sig));
    }
}

void vjp_Mish(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (X->requires_grad()) {
        Tensor x = X->value;
        Tensor sp = OwnTensor::log(1.0f + OwnTensor::exp(x, ag::current_stream()), ag::current_stream());
        Tensor tsp = OwnTensor::tanh(sp);
        Tensor sig = 1.0f / (1.0f + OwnTensor::exp(x * -1.0f, ag::current_stream()));
        X->grad += gy * (tsp + x * (1.0f - tsp * tsp) * sig);
    }
}

// Parametric Activations ------
void vjp_LeakyRelu(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    Node* Alpha = n->inputs[1].get();
    if (X->requires_grad()) {
        float alpha = Alpha->value.to_cpu().data<float>()[0];
        Tensor mask = OwnTensor::Tensor::zeros(X->value.shape(), ag::options(X->value));
        dispatch_by_dtype(X->value.dtype(), [&](auto dummy){
            using T = decltype(dummy);
            const T* x_ptr = X->value.data<T>();
            T* m_ptr = mask.data<T>();
            for(int64_t i=0; i<X->value.numel(); ++i) {
                if constexpr (std::is_same_v<T, OwnTensor::complex32_t> || 
                              std::is_same_v<T, OwnTensor::complex64_t> || 
                              std::is_same_v<T, OwnTensor::complex128_t>) {
                    m_ptr[i] = (x_ptr[i].real() > 0) ? T(1.0f) : T(alpha);
                } else {
                    m_ptr[i] = (x_ptr[i] > T(0)) ? T(1.0f) : T(alpha);
                }
            }
        });
        X->grad += gy * mask;
    }
}
// Specialized activations -----------
void vjp_Gaus(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (X->requires_grad()) {
        X->grad += gy * n->value * (X->value * -2.0f);
    }
}

void vjp_Parcon(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (X->requires_grad()) {
        X->grad += gy * (2.0f - X->value * 2.0f);
    }
}

void vjp_LiSHT(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (X->requires_grad()) {
        Tensor x = X->value;
        Tensor tx = OwnTensor::tanh(x);
        X->grad += gy * (tx + x * (1.0f - tx * tx));
    }
}
// Standard Attention ---------
void vjp_Attention(Node* n, const Tensor& gy){
    throw std::runtime_error("VJP for Attention not implemented yet!");
}

// Gated activation -----------------
void vjp_SWIGLU(Node* n, const Tensor& gy){
    throw std::runtime_error("VJP for SWIGLU not implemented yet!");
}

//Leaf -----------
void vjp_Leaf(Node*, const Tensor&){ /* no-op */ }

//Unary Mathematical Functions ------------------
void vjp_Exp(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (X->requires_grad()) X->grad += gy * n->value;
}

void vjp_Log(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (X->requires_grad()) X->grad += gy / X->value;
}


void vjp_Sqrt(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (X->requires_grad()) X->grad += gy * 0.5f / n->value;
}

void vjp_Reciprocal(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (X->requires_grad()) X->grad -= gy / (X->value * X->value);
}

void vjp_Sign(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (X->requires_grad()) X->grad += gy * 0.0f;
}
void vjp_Abs(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (X->requires_grad()) {
        const float epsilon = 1e-9f;
        X->grad += gy * (X->value / (n->value + epsilon));
    }
}
void vjp_Pow(Node* n, const Tensor& gy){
    Node* A_node = n->inputs[0].get();
    Node* B_node = n->inputs[1].get();
    const Tensor& A = A_node->value;
    const Tensor& B = B_node->value;
    if (A_node->requires_grad()) {
        A_node->grad += reduce_for_broadcast(gy * B * OwnTensor::exp((B - 1.0f) * OwnTensor::log(A, ag::current_stream()), ag::current_stream()), A);
    }
    if (B_node->requires_grad()) {
        B_node->grad += reduce_for_broadcast(gy * n->value * OwnTensor::log(A, ag::current_stream()), B);
    }
}

//Core Matrix Operations ------------------------
void vjp_MatMul(Node* n, const Tensor& gy){
    Node* A = n->inputs[0].get();
    Node* B = n->inputs[1].get();
    if (A->requires_grad()) A->grad += OwnTensor::matmul(gy, B->value.t());
    if (B->requires_grad()) B->grad += OwnTensor::matmul(A->value.t(), gy);
}

void vjp_Transpose(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (X->requires_grad()) X->grad += gy.t();
}

//Fused Operations (better performance, fewer memory accesses) -------------

void vjp_Linear(Node* n, const Tensor& gy){
    Node* X_node = n->inputs[0].get();
    Node* W_node = n->inputs[1].get();
    Node* b_node = n->inputs[2].get();
    if (X_node->requires_grad()) X_node->grad += OwnTensor::matmul(gy, W_node->value);
    if (W_node->requires_grad()) W_node->grad += OwnTensor::matmul(gy.t(), X_node->value);
    if (b_node->requires_grad()) b_node->grad += OwnTensor::reduce_sum(gy, {0}, false);
}

void vjp_FMA(Node* n, const Tensor& gy){
    Node* A = n->inputs[0].get();
    Node* B = n->inputs[1].get();
    Node* C = n->inputs[2].get();
    if (A->requires_grad()) A->grad += OwnTensor::matmul(gy, B->value.t());
    if (B->requires_grad()) B->grad += OwnTensor::matmul(A->value.t(), gy);
    if (C->requires_grad()) C->grad += gy;
}

//Classification losses ---------------
void vjp_CeWithLogits(Node* n, const Tensor& gy){
    Node* Z_node = n->inputs[0].get();
    Node* Y_node = n->inputs[1].get();
    const Tensor& Z = Z_node->value;
    const Tensor& Y = Y_node->value;
    const float inv_batch_size = 1.0f / static_cast<float>(Z.shape().dims[0]);
    Tensor max_val = OwnTensor::reduce_max(Z, {-1}, true);
    Tensor z_shifted = Z - max_val;
    Tensor exp_z = OwnTensor::exp(z_shifted, ag::current_stream());
    Tensor sum_exp_z = OwnTensor::reduce_sum(exp_z, {-1}, true);
    Tensor softmax_z = exp_z / sum_exp_z;
    if (Z_node->requires_grad()) {
        Z_node->grad += gy * (softmax_z - Y) * inv_batch_size;
    }
}

void vjp_KLDivergence(Node* n, const Tensor& gy){
    Node* Z_node = n->inputs[0].get();
    Node* Y_node = n->inputs[1].get();
    const Tensor& Z = Z_node->value;
    const Tensor& Y = Y_node->value;
    const float inv_batch_size = 1.0f / static_cast<float>(Z.shape().dims[0]);
    Tensor max_val = OwnTensor::reduce_max(Z, {-1}, true);
    Tensor z_shifted = Z - max_val;
    Tensor exp_z = OwnTensor::exp(z_shifted, ag::current_stream());
    Tensor sum_exp_z = OwnTensor::reduce_sum(exp_z, {-1}, true);
    Tensor softmax_z = exp_z / sum_exp_z;
    if (Z_node->requires_grad()) {
        Z_node->grad += gy * (softmax_z - Y) * inv_batch_size;
    }
    if (Y_node->requires_grad()) {
        Tensor log_Y = OwnTensor::log(Y + 1e-9f, ag::current_stream());
        Tensor log_softmax_z = z_shifted - OwnTensor::log(sum_exp_z, ag::current_stream());
        Y_node->grad += gy * (log_Y + 1.0f - log_softmax_z) * inv_batch_size;
    }
}

//Regression Losses --------------
void vjp_MSELoss(Node* n, const Tensor& gy){
    Node* Z_node = n->inputs[0].get();
    Node* Y_node = n->inputs[1].get();
    const float gy_scalar = gy.to_cpu().data<float>()[0];
    const float scale = 2.0f / static_cast<float>(Z_node->value.numel());
    Tensor diff = Z_node->value - Y_node->value;
    if (Z_node->requires_grad()) Z_node->grad += diff * (gy_scalar * scale);
    if (Y_node->requires_grad()) Y_node->grad -= diff * (gy_scalar * scale);
}

void vjp_MAELoss(Node* n, const Tensor& gy){
    Node* Z_node = n->inputs[0].get();
    Node* Y_node = n->inputs[1].get();
    const float inv_N = 1.0f / static_cast<float>(Z_node->value.numel());
    const float epsilon = 1e-9f;
    Tensor diff = Z_node->value - Y_node->value;
    Tensor sign_diff = diff / (OwnTensor::abs(diff, ag::current_stream()) + epsilon);
    if (Z_node->requires_grad()) Z_node->grad += gy * sign_diff * inv_N;
    if (Y_node->requires_grad()) Y_node->grad -= gy * sign_diff * inv_N;
}

//Layer Normalization ------------
void vjp_LayerNorm(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (X->requires_grad()) {
        const Tensor& x = X->value;
        const Tensor& var = *n->tape[0];
        const Tensor& mean = *n->tape[1];
        Tensor std_inv = 1.0f / OwnTensor::sqrt(var + 1e-5f, ag::current_stream());
        Tensor x_hat = (x - mean) * std_inv;
        int64_t D = x.shape().dims.back();
        Tensor d_xhat = gy * std_inv;
        Tensor d_var = OwnTensor::reduce_sum(gy * (x - mean) * -0.5f * (std_inv * std_inv * std_inv), {-1}, true);
        Tensor d_mean = OwnTensor::reduce_sum(gy * (std_inv * -1.0f), {-1}, true) + d_var * OwnTensor::reduce_sum((x - mean) * -2.0f, {-1}, true) / static_cast<float>(D);
        X->grad += d_xhat + d_var * (x - mean) * (2.0f / static_cast<float>(D)) + d_mean / static_cast<float>(D);
    }
}

//RMS Normalization -------------
void vjp_RMSNorm(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (X->requires_grad()) {
        const Tensor& x = X->value;
        const Tensor& rsqrt_var = *n->tape[0];
        int64_t D = x.shape().dims.back();
        Tensor dot = OwnTensor::reduce_sum(gy * x, {-1}, true);
        X->grad += rsqrt_var * (gy - x * (rsqrt_var * rsqrt_var) * dot / static_cast<float>(D));
    }
}

void vjp_RealRMSNorm(Node* n, const Tensor& gy){
    throw std::runtime_error("VJP for RealRMSNorm not implemented yet!");
}

//Dynamic Normalization --------------
void vjp_Dyntanh(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    Node* A = n->inputs[1].get();
    Node* B = n->inputs[2].get();
    Node* G = n->inputs[3].get();
    const Tensor& h = *n->tape[0];
    Tensor th = OwnTensor::tanh(h);
    Tensor d_th = 1.0f - th * th;
    if (X->requires_grad()) X->grad += gy * G->value * d_th * A->value;
    if (A->requires_grad()) A->grad += OwnTensor::reduce_sum(gy * G->value * d_th * X->value, {}, false);
    if (B->requires_grad()) B->grad += OwnTensor::reduce_sum(gy, {}, false);
    if (G->requires_grad()) G->grad += OwnTensor::reduce_sum(gy * th, {}, false);
}

//Global Reductions -------------------
void vjp_Sum(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (X->requires_grad()) X->grad += gy;
}
void vjp_MeanAll(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (X->requires_grad()) {
        float scale = 1.0f / static_cast<float>(X->value.numel());
        X->grad += gy * scale;
    }
}
//Row-wise Reductions ------------------------
void vjp_RowSum(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (X->requires_grad()) X->grad += gy;
}

void vjp_RowMax(Node* n, const Tensor& gy){
    throw std::runtime_error("VJP for RowMax not implemented yet!");
}

//Softmax Family ---------------
void vjp_SoftmaxRow(Node* n, const Tensor& gy){
    Node* Z = n->inputs[0].get();
    if (Z->requires_grad()) {
        const Tensor& y = n->value;
        Tensor dot = OwnTensor::reduce_sum(y * gy, {-1}, true);
        Z->grad += y * (gy - dot);
    }
}

void vjp_LogSumExpRow(Node* n, const Tensor& gy){
    Node* Z = n->inputs[0].get();
    if (Z->requires_grad()) {
        const Tensor& z_val = Z->value;
        Tensor max_val = OwnTensor::reduce_max(z_val, {-1}, true);
        Tensor exp_z = OwnTensor::exp(z_val - max_val, ag::current_stream());
        Tensor sum_exp_z = OwnTensor::reduce_sum(exp_z, {-1}, true);
        Tensor softmax_z = exp_z / sum_exp_z;
        Z->grad += gy * softmax_z;
    }
}

//Trigonometric Functions --------------------
void vjp_Sin(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (X->requires_grad()) X->grad += gy * OwnTensor::cos(X->value);
}

void vjp_Cos(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (X->requires_grad()) X->grad -= gy * OwnTensor::sin(X->value);
}

void vjp_Tan(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (X->requires_grad()) {
        Tensor c = OwnTensor::cos(X->value);
        X->grad += gy / (c * c);
    }
}

//Hyperbolic Functions ------------------
void vjp_Cosh(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (X->requires_grad()) X->grad += gy * OwnTensor::sinh(X->value);
}
void vjp_Sinh(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (X->requires_grad()) X->grad += gy * OwnTensor::cosh(X->value);
}

//Inverse Trigonometric Functions --------------
void vjp_Asin(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (X->requires_grad()) X->grad += gy / OwnTensor::sqrt(1.0f - X->value * X->value, ag::current_stream());
}

void vjp_Acos(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (X->requires_grad()) X->grad -= gy / OwnTensor::sqrt(1.0f - X->value * X->value, ag::current_stream());
}

void vjp_Atan(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (X->requires_grad()) X->grad += gy / (1.0f + X->value * X->value);
}

//Inverse Hyperbolic Trigonometric Functions ----------------
void vjp_ASinh(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (X->requires_grad()) X->grad += gy / OwnTensor::sqrt(X->value * X->value + 1.0f, ag::current_stream());
}

void vjp_ACosh(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (X->requires_grad()) X->grad += gy / OwnTensor::sqrt(X->value * X->value - 1.0f, ag::current_stream());
}

void vjp_ATanh(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (X->requires_grad()) X->grad += gy / (1.0f - X->value * X->value);
}


} // namespace detail

VjpFn vjp_lookup(Op op){
    switch(op){
#define OP(name, arity, str) case Op::name: return &detail::vjp_##name;
#include "ad/detail/ops.def"
#undef OP
        default: return nullptr;
    }
}

} // namespace ag
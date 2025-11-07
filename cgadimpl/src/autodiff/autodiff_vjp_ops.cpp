// ====================================================================
// FILE: cgadimpl/src/autodiff/autodiff_vjp_ops.cpp (GPU-Aware Version)
// ====================================================================

#include "ad/detail/autodiff_ops.hpp"
#include "ad/runtime.hpp"
#include <cmath>
#include <stdexcept> // Required for std::runtime_error

namespace ag {
namespace detail{



// ----- elementwise binary -----
// Correct: Accumulates gradient for both parents.
void vjp_Add(Node* n, const Tensor& gy){
    if (n->inputs[0]->requires_grad()) n->inputs[0]->grad += gy;
    if (n->inputs[1]->requires_grad()) n->inputs[1]->grad += gy;
}

// Correct: Accumulates +gy for A and -gy for B.
void vjp_Sub(Node* n, const Tensor& gy){
    if (n->inputs[0]->requires_grad()) n->inputs[0]->grad += gy;
    if (n->inputs[1]->requires_grad()) n->inputs[1]->grad += (gy * -1.0f);
}

// Correct: Chain rule for multiplication.
void vjp_Mul(Node* n, const Tensor& gy){
    if (n->inputs[0]->requires_grad()) n->inputs[0]->grad += (gy * n->inputs[1]->value);
    if (n->inputs[1]->requires_grad()) n->inputs[1]->grad += (gy * n->inputs[0]->value);
}

// ----- elementwise trinary & matmul -----
// ===================================================================
// vjp_FMA
// ===================================================================
void vjp_FMA(Node* n, const Tensor& gy){
    Node* A = n->inputs[0].get();
    Node* B = n->inputs[1].get();
    Node* C = n->inputs[2].get();
    
    const Tensor& At = A->value;
    const Tensor& Bt = B->value;

    // The OwnTensor operators handle device, stream, and broadcasting automatically.
    if (A->requires_grad()){
        A->grad += OwnTensor::matmul(gy, Bt.t());
    }
    if (B->requires_grad()){
        B->grad += OwnTensor::matmul(At.t(), gy);
    }
    if (C->requires_grad()) {
        C->grad += gy;
    }
}

// ----- Normalization Layers -----
// // ===================================================================
// // vjp_LayerNorm
// // ===================================================================
// void vjp_LayerNorm(Node* n, const Tensor& gy){
//     Node* x_node = n->inputs[0].get();
//     if (!x_node->requires_grad()) return;

//     const Tensor& x = x_node->value;
//     const Tensor& variance = *n->tape[0];
//     const Tensor& mean = *n->tape[1];

//     const float N = static_cast<float>(x.shape().dims.back());
    
//     Tensor std_dev_inv = 1.0f / OwnTensor::sqrt(variance + 1e-5f, ag::current_stream());
//     Tensor x_normalized = (x - mean) * std_dev_inv;

//     // --- Start VJP Calculation ---
    
//     // VJP of the normalization: dL/dx_normalized
//     // Simplified from your original code. Let's use a more standard formulation.
//     // Ref: https://github.com/karpathy/makemore/blob/master/makemore.py (search for layernorm backward)
    
//     // 1. Gradient of the normalized output
//     Tensor d_x_normalized = gy;
    
//     // 2. Gradient of the variance
//     // dL/dvar = sum(dL/dx_norm * (x - mu) * -0.5 * (var + eps)^-1.5, axis=-1)
//     Tensor d_variance = OwnTensor::reduce_sum(
//         d_x_normalized * (x - mean) * -0.5f * OwnTensor::pow(variance + 1e-5f, -1.5f, ag::current_stream()), 
//         {-1}, 
//         true
//     );
    
//     // 3. Gradient of the mean
//     // dL/dmu = sum(dL/dx_norm * -1/std, axis=-1) + dL/dvar * sum(-2*(x-mu)/N, axis=-1)
//     Tensor d_mean = OwnTensor::reduce_sum(d_x_normalized * -1.0f * std_dev_inv, {-1}, true) + 
//                   d_variance * OwnTensor::reduce_sum(-2.0f * (x - mean) / N, {-1}, true);

//     // 4. Gradient of the input x
//     // dL/dx = dL/dx_norm * 1/std + dL/dvar * 2*(x-mu)/N + dL/dmu * 1/N
//     Tensor dx = (d_x_normalized * std_dev_inv) + 
//                 (d_variance * 2.0f * (x - mean) / N) +
//                 (d_mean / N);

//     x_node->grad += dx;
// }

// ===================================================================
// vjp_LayerNorm
// ===================================================================
void vjp_LayerNorm(Node* n, const Tensor& gy){
    Node* x = n->inputs[0].get();
    if (!x->requires_grad()) return;

    // Get the number of features from the last dimension
    const float N = static_cast<float>(x->value.shape().dims.back());
    
    const Tensor& variance = *n->tape[0];
    const Tensor& mean = *n->tape[1];

    Tensor std_dev = OwnTensor::sqrt(variance + 1e-5f, ag::current_stream()); // Use OwnTensor::sqrt
    Tensor xmu = x->value - mean;

    // Use reduce_sum for row-wise summation
    Tensor grad_sum = OwnTensor::reduce_sum(gy, {-1}, true);
    Tensor grad_dot_xmu = OwnTensor::reduce_sum(gy * xmu, {-1}, true);

    Tensor term1 = gy * N;
    Tensor term2 = term1 - grad_sum;
    Tensor term3 = term2 - (xmu * (grad_dot_xmu / (variance + 1e-5f)));
    Tensor dx = term3 / (std_dev * N);
    
    x->grad += dx;
}

// ===================================================================
// vjp_RMSNorm
// ===================================================================
void vjp_RMSNorm(Node* n, const Tensor& gy){
    Node* x = n->inputs[0].get();
    if (!x->requires_grad()) return;

    const Tensor& rms = *n->tape[0]; // rsqrt(variance + epsilon)
    const Tensor& y_normalized = *n->tape[1]; // x * rsqrt(...)
    
    const float inv_N = 1.0f / static_cast<float>(x->value.shape().dims.back());

    Tensor dot = OwnTensor::reduce_sum(gy * y_normalized, {-1}, true);
    
    // grad_x = rsqrt * (gy - y * dot)
    Tensor grad_x = rms * (gy - y_normalized * dot);

    x->grad += grad_x;
}

// ===================================================================
// vjp_RealLayerNorm
// ===================================================================
void vjp_RealLayerNorm(Node* n, const Tensor& gy){
    Node* x = n->inputs[0].get();
    Node* g = n->inputs[1].get(); // Gain
    Node* b = n->inputs[2].get(); // Bias
    
    const float N = static_cast<float>(x->value.shape().dims.back());

    const Tensor& variance = *n->tape[0];
    const Tensor& mean = *n->tape[1];
    const Tensor& x_normalized = *n->tape[2];
    
    Tensor std_dev = OwnTensor::sqrt(variance + 1e-5f, ag::current_stream());
    Tensor xmu = x->value - mean;

    // VJP for the core normalization part (same as LayerNorm)
    Tensor grad_sum = OwnTensor::reduce_sum(gy, {-1}, true);
    Tensor grad_dot_xmu = OwnTensor::reduce_sum(gy * x_normalized, {-1}, true);

    Tensor term1 = gy * N;
    Tensor term2 = term1 - grad_sum;
    Tensor term3 = term2 - (x_normalized * grad_dot_xmu);
    Tensor dx_normalized = term3 / N;

    // VJP for the affine transformation (scale and shift)
    if (g->requires_grad()) {
        g->grad += OwnTensor::reduce_sum(gy * x_normalized, {-1});
    }
    if (b->requires_grad()) {
        b->grad += OwnTensor::reduce_sum(gy, {-1});
    }
    if (x->requires_grad()) {
        x->grad += (g->value / std_dev) * dx_normalized;
    }
}

// ----- Attention Mechanisms -----
// ===================================================================
// vjp_Attention
// ===================================================================
void vjp_Attention(Node* n, const Tensor& gy){
    Node* A = n->inputs[0].get();
    Node* B = n->inputs[1].get();
    Node* C = n->inputs[2].get();
    Node* D = n->inputs[3].get();
    
    // Tensors from the forward pass, saved on the tape
    const Tensor& q = *n->tape[0];
    const Tensor& k = *n->tape[1];
    const Tensor& v = *n->tape[2];
    const Tensor& s = *n->tape[3]; // The softmax output

    float scale = 1.0f / std::sqrt(static_cast<float>(k.shape().dims.back()));

    // All ops below will now use the stream-aware OwnTensor API
    Tensor dL_ds = OwnTensor::matmul(gy, v.t());
    Tensor dL_dv = OwnTensor::matmul(s.t(), gy);
    
    // VJP of softmax: s * (dL_ds - row_sum(s * dL_ds))
    Tensor dot = OwnTensor::reduce_sum(s * dL_ds, {-1}, true);
    Tensor dL_dg = s * (dL_ds - dot);
    
    // Propagate gradients back through the Q, K projections
    Tensor dL_dq = OwnTensor::matmul(dL_dg, k);
    Tensor dL_dk = OwnTensor::matmul(dL_dg.t(), q);

    // Propagate gradients to the weight matrices and the input A
    if (B->requires_grad()) {
        B->grad += OwnTensor::matmul(A->value.t(), dL_dq) * scale;
    }
    if (C->requires_grad()) {
        C->grad += OwnTensor::matmul(A->value.t(), dL_dk) * scale;
    }
    if (D->requires_grad()) {
        D->grad += OwnTensor::matmul(A->value.t(), dL_dv);
    }
    if (A->requires_grad()) {
        Tensor dL_dA_q = OwnTensor::matmul(dL_dq, B->value);
        Tensor dL_dA_k = OwnTensor::matmul(dL_dk, C->value);
        Tensor dL_dA_v = OwnTensor::matmul(dL_dv, D->value);
        A->grad += (dL_dA_q * scale) + (dL_dA_k * scale) + dL_dA_v;
    }
}

void vjp_AlibiAttention(Node* n, const Tensor& gy){
    // Assuming same VJP as Attention for now
    vjp_Attention(n, gy);
}

// ===================================================================
// vjp_SWIGLU
// ===================================================================
void vjp_SWIGLU(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    Node* A = n->inputs[1].get();
    Node* B = n->inputs[2].get();
    Node* C = n->inputs[3].get();
    Node* D = n->inputs[4].get();

    // Recompute intermediates using OwnTensor API
    Tensor y = OwnTensor::matmul(X->value, A->value.t()) + B->value;
    Tensor h = OwnTensor::matmul(X->value, C->value.t()) + D->value;

    // --- Re-implement sigmoid and its derivative using OwnTensor ops ---
    Tensor sig_y = 1.0f / (1.0f + OwnTensor::exp(y * -1.0f));
    Tensor swish_y = y * sig_y;
    Tensor swish_grad = sig_y + y * sig_y * (1.0f - sig_y);
    // ---

    // Propagate gradients
    Tensor dL_dh = swish_y * gy;
    Tensor dL_dy = h * swish_grad * gy;

    if (D->requires_grad()) D->grad += dL_dh;
    if (C->requires_grad()) C->grad += OwnTensor::matmul(dL_dh.t(), X->value);
    
    if (B->requires_grad()) B->grad += dL_dy;
    if (A->requires_grad()) A->grad += OwnTensor::matmul(dL_dy.t(), X->value);
    
    if (X->requires_grad()) {
        X->grad += OwnTensor::matmul(dL_dh, C->value) + OwnTensor::matmul(dL_dy, A->value);
    }
}
// ===================================================================
// vjp_Relu
// ===================================================================
void vjp_Relu(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad()) return;
    const float epsilon = 1e-9f;
    Tensor sign_X = X->value / (OwnTensor::abs(X->value, ag::current_stream()) + epsilon);
    Tensor mask = (sign_X + OwnTensor::abs(sign_X, ag::current_stream())) * 0.5f;
    X->grad += gy * mask;
}

// ===================================================================
// vjp_Exp
// ===================================================================

void vjp_Exp(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad()) return;

    // The VJP for exp(x) is gy * exp(x). The forward pass output is exp(x).
    // This uses the stream-aware OwnTensor operator '*' for both CPU and GPU.
    X->grad += gy * n->value;
}

// ===================================================================
// vjp_Log
// ===================================================================

void vjp_Log(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad()) return;

    // The VJP for log(x) is gy / x.
    // This uses the stream-aware OwnTensor operator '/' for both CPU and GPU.
    X->grad += gy / X->value;
}


// ===================================================================
// vjp_GCU
// ===================================================================
void vjp_GCU(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad()) return;

    // VJP is gy * (cos(x) - x * sin(x))
    // All ops are from OwnTensor and are stream-aware.
    Tensor d_gcu = OwnTensor::cos(X->value) - (X->value * OwnTensor::sin(X->value));
    X->grad += gy * d_gcu;
}

// ===================================================================
// vjp_Mish
// ===================================================================
void vjp_Mish(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad()) return;

    // Re-calculate intermediates needed for the derivative
    // softplus(x) = log(1 + exp(x))
    Tensor sp = OwnTensor::log(1.0f + OwnTensor::exp(X->value));
    Tensor tanh_sp = OwnTensor::tanh(sp);
    
    // sigmoid(x) = 1 / (1 + exp(-x))
    Tensor sig_x = 1.0f / (1.0f + OwnTensor::exp(X->value * -1.0f));

    // The derivative of mish
    Tensor d_mish = tanh_sp + X->value * (1.0f - (tanh_sp * tanh_sp)) * sig_x;
    
    // Apply the chain rule
    X->grad += gy * d_mish;
}

// ===================================================================
// vjp_Tanh
// ===================================================================
void vjp_Tanh(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad()) return;

    // VJP is gy * (1 - tanh(x)^2)
    // Here, t = n->value is the result of the forward tanh(x)
    const Tensor& t = n->value;
    X->grad += gy * (1.0f - (t * t));
}

// ===================================================================
// vjp_Sigmoid
// ===================================================================
void vjp_Sigmoid(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad()) return;

    // VJP is gy * (sigmoid(x) * (1 - sigmoid(x)))
    // Here, s = n->value is the result of the forward sigmoid(x)
    const Tensor& s = n->value;
    X->grad += gy * (s * (1.0f - s));
}


// ===================================================================
// vjp_Softplus
// ===================================================================
void vjp_Softplus(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad()) return;

    // VJP is gy * sigmoid(x)
    // sigmoid(x) = 1 / (1 + exp(-x))
    Tensor d_softplus = 1.0f / (1.0f + OwnTensor::exp(X->value * -1.0f));
    
    X->grad += gy * d_softplus;
}


// ===================================================================
// vjp_Gaus
// ===================================================================
void vjp_Gaus(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad()) return;

    // VJP is gy * (-2 * x * exp(-x^2))
    // We can reuse the forward pass output, n->value, which is exp(-x^2).
    X->grad += gy * -2.0f * X->value * n->value;
}

// ===================================================================
// vjp_Transpose
// ===================================================================
void vjp_Transpose(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad()) return;

    // The VJP is just the transpose of the gradient.
    X->grad += gy.t();
}

// ===================================================================
// vjp_SiLU
// ===================================================================
void vjp_SiLU(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad()) return;

    // Recompute sigmoid(x)
    Tensor s = 1.0f / (1.0f + OwnTensor::exp(X->value * -1.0f));
    
    // Derivative of SiLU
    Tensor d_silu = s * (1.0f + X->value * (1.0f - s));
    
    X->grad += gy * d_silu;
}

// ===================================================================
// vjp_Parcon
// ===================================================================
void vjp_Parcon(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad()) return;

    // VJP is gy * (2 - 2*x)
    X->grad += gy * (2.0f - 2.0f * X->value);
}

// ===================================================================
// vjp_LiSHT
// ===================================================================
void vjp_LiSHT(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad()) return;

    // Recompute tanh(x)
    Tensor th_x = OwnTensor::tanh(X->value);
    
    // Derivative is tanh(x) + x * sech(x)^2, which is tanh(x) + x * (1 - tanh(x)^2)
    Tensor d_lisht = th_x + X->value * (1.0f - (th_x * th_x));
    
    X->grad += gy * d_lisht;
}

// ===================================================================
// vjp_GELU
// ===================================================================
void vjp_GELU(Node* n, const Tensor& gy){
    Node* X_node = n->inputs[0].get();
    if (!X_node->requires_grad()) return;
    const Tensor& x = X_node->value;

    // Constants for the GELU approximation's derivative
    const float c1 = 0.7978845608f; // sqrt(2.0f / M_PI)
    const float c2 = 0.044715f;

    // Recompute intermediates needed for the derivative
    Tensor x2 = x * x;
    Tensor x3 = x2 * x;
    Tensor u = (x + x3 * c2) * c1;
    Tensor th_u = OwnTensor::tanh(u);
    
    // Compute du/dx = c1 * (1 + 3 * c2 * x^2)
    Tensor du_dx = (1.0f + (x2 * (3.0f * c2))) * c1;

    // Compute the full GELU derivative
    Tensor d_gelu = (1.0f + th_u) * 0.5f + (x * (1.0f - (th_u * th_u)) * du_dx) * 0.5f;

    // Apply the chain rule
    X_node->grad += gy * d_gelu;
}
// ===================================================================
// vjp_LeakyRelu
// ===================================================================
void vjp_LeakyRelu(Node* n, const Tensor& gy){
    Node* X_node = n->inputs[0].get();
    if (!X_node->requires_grad()) return;
    const Tensor& x = X_node->value;
    
    // Get alpha from the second input node
    Node* A_node = n->inputs[1].get();
    // Use .data<T>()[0] to get the scalar value from the 1x1 tensor
    float alpha = A_node->value.data<float>()[0]; 

    // --- Create the Leaky ReLU derivative mask using pure arithmetic ---
    // The mask should be 1 where x > 0 and alpha where x <= 0.
    
    // 1. Create a mask of 1s and 0s for where x > 0
    const float epsilon = 1e-9f;
    Tensor sign_x = x / (OwnTensor::abs(x, ag::current_stream()) + epsilon);
    Tensor mask_pos = (sign_x + OwnTensor::abs(sign_x, ag::current_stream())) * 0.5f; // relu(sign(x))
    
    // 2. Create a mask of 1s and 0s for where x <= 0
    Tensor mask_neg = 1.0f - mask_pos;
    
    // 3. Combine them: (mask_pos * 1.0) + (mask_neg * alpha)
    Tensor d_leaky = mask_pos + (mask_neg * alpha);

    // Apply the chain rule
    X_node->grad += gy * d_leaky;
}

// ===================================================================
// vjp_MatMul
// ===================================================================
void vjp_MatMul(Node* n, const Tensor& gy){
    Node* A_node = n->inputs[0].get();
    Node* B_node = n->inputs[1].get();
    const Tensor& A = A_node->value;
    const Tensor& B = B_node->value;

    // VJP for A: dL/dA = dL/dY @ B^T
    if (A_node->requires_grad()) {
        A_node->grad += OwnTensor::matmul(gy, B.t());
    }

    // VJP for B: dL/dB = A^T @ dL/dY
    if (B_node->requires_grad()) {
        B_node->grad += OwnTensor::matmul(A.t(), gy);
    }
}

// ===================================================================
// vjp_Dyntanh
// ===================================================================
void vjp_Dyntanh(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get(); 
    Node* A = n->inputs[1].get(); 
    Node* B = n->inputs[2].get(); 
    Node* G = n->inputs[3].get();
    
    // The tape stores h = a*x from the forward pass.
    const Tensor& h = *(n->tape.back());
    Tensor th_h = OwnTensor::tanh(h);
    
    // Derivative of tanh(h) is 1 - tanh(h)^2
    Tensor d_tanh = 1.0f - (th_h * th_h);
    
    if (G->requires_grad()) {
        G->grad += gy * th_h;
    }
    if (B->requires_grad()) {
        B->grad += gy;
    }
    if (A->requires_grad()) {
        // Chain rule: gy * g * d_tanh * x
        A->grad += gy * G->value * d_tanh * X->value;
    }
    if (X->requires_grad()) {
        // Chain rule: gy * g * d_tanh * a
        X->grad += gy * G->value * d_tanh * A->value;
    }
}

// ----- Reductions -----
// ===================================================================
// vjp_Sum
// ===================================================================
void vjp_Sum(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad()) return;

    // `gy` is a 1x1 scalar tensor. The '+' operator will automatically
    // broadcast it to the shape of X->grad.
    X->grad += gy;
}

// ===================================================================
// vjp_RowSum
// ===================================================================
void vjp_RowSum(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad()) return;

    // `gy` has shape [B, 1]. The '+' operator will automatically
    // broadcast it to the shape of X->grad, which is [B, C].
    X->grad += gy;
}

// ===================================================================
// vjp_RowMax
// ===================================================================
void vjp_RowMax(Node* n, const Tensor& gy){
    // To implement this, we need to know the *index* of the max element in each row.
    // This requires an `argmax` function which is not available in OwnTensor.
    // The VJP would look like:
    //   Tensor indices = OwnTensor::argmax(X->value, /*axis=*/-1);
    //   X->grad.scatter_add_(/*axis=*/-1, indices, gy);
    throw std::runtime_error("VJP for RowMax cannot be implemented without argmax or comparison ops in the tensor library.");
}

// ===================================================================
// vjp_MeanAll
// ===================================================================

void vjp_MeanAll(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad()) return;
    
    // The gradient needs to be scaled by 1/N.
    float scale = 1.0f / static_cast<float>(X->value.numel());
    
    // `gy` is a scalar. `gy * scale` is also a scalar.
    // The `+=` operator will broadcast this scalar across the entire gradient tensor.
    X->grad += gy * scale;
}

// ===================================================================
// vjp_SoftmaxRow
// ===================================================================
void vjp_SoftmaxRow(Node* n, const Tensor& gy){
    Node* Z = n->inputs[0].get();
    if (!Z->requires_grad()) return;

    // y is the output of the softmax, which is stored in the node's value.
    const Tensor& y = n->value;

    // Calculate the dot product along the rows.
    // This needs to be a sum, not a matmul.
    Tensor dot = OwnTensor::reduce_sum(y * gy, {-1}, true);

    // The += operator will broadcast 'dot' correctly.
    Z->grad += y * (gy - dot);
}

// ===================================================================
// vjp_LogSumExpRow
// ===================================================================
void vjp_LogSumExpRow(Node* n, const Tensor& gy){
    Node* Z = n->inputs[0].get();
    if (!Z->requires_grad()) return;

    // --- Re-implement softmax using OwnTensor ops ---
    // This is the derivative of logsumexp.
    const Tensor& z_val = Z->value;
    Tensor max_val = OwnTensor::reduce_max(z_val, {-1}, true);
    Tensor exp_z = OwnTensor::exp(z_val - max_val);
    Tensor sum_exp_z = OwnTensor::reduce_sum(exp_z, {-1}, true);
    Tensor softmax_z = exp_z / sum_exp_z;
    // ---
    
    // The VJP is gy * softmax(z). The += operator will handle broadcasting.
    Z->grad += gy * softmax_z;
}

// ===================================================================
// vjp_CeWithLogits
// ===================================================================
void vjp_CeWithLogits(Node* n, const Tensor& gy){
    Node* Z_node = n->inputs[0].get();
    Node* Y_node = n->inputs[1].get();
    const Tensor& Z = Z_node->value;
    const Tensor& Y = Y_node->value;
    
    // Batch size is the size of the first dimension
    const float inv_batch_size = 1.0f / static_cast<float>(Z.shape().dims[0]);

    // Re-calculate stable softmax and log_softmax
    Tensor max_val = OwnTensor::reduce_max(Z, {-1}, true);
    Tensor z_shifted = Z - max_val;
    Tensor exp_z = OwnTensor::exp(z_shifted);
    Tensor sum_exp_z = OwnTensor::reduce_sum(exp_z, {-1}, true);
    Tensor softmax_z = exp_z / sum_exp_z;
    Tensor log_softmax_z = z_shifted - OwnTensor::log(sum_exp_z);
    
    if (Z_node->requires_grad()) {
        // gZ = (softmax(Z) - Y) / batch_size
        Tensor gZ = (softmax_z - Y) * inv_batch_size;
        // The `gy` for a loss function is typically a scalar. The operators will broadcast it.
        Z_node->grad += gy * gZ;
    }
    if (Y_node->requires_grad()) {
        // gY = -log_softmax(Z) / batch_size
        Tensor gY = log_softmax_z * (-1.0f * inv_batch_size);
        Y_node->grad += gy * gY;
    }
}

// ===================================================================
// vjp_KLDivergence
// ===================================================================
void vjp_KLDivergence(Node* n, const Tensor& gy){
    Node* Z_node = n->inputs[0].get();
    Node* Y_node = n->inputs[1].get();
    const Tensor& Z = Z_node->value;
    const Tensor& Y = Y_node->value;
    
    const float inv_batch_size = 1.0f / static_cast<float>(Z.shape().dims[0]);

    // Re-calculate stable softmax and log_softmax
    Tensor max_val = OwnTensor::reduce_max(Z, {-1}, true);
    Tensor z_shifted = Z - max_val;
    Tensor exp_z = OwnTensor::exp(z_shifted);
    Tensor sum_exp_z = OwnTensor::reduce_sum(exp_z, {-1}, true);
    Tensor softmax_z = exp_z / sum_exp_z;
    Tensor log_softmax_z = z_shifted - OwnTensor::log(sum_exp_z);

    if (Z_node->requires_grad()) {
        // gZ = softmax(Z) - Y, scaled by batch size
        Tensor gZ = (softmax_z - Y) * inv_batch_size;
        Z_node->grad += gy * gZ;
    }
    if (Y_node->requires_grad()) {
        // gY = log(Y) + 1 - log_softmax(Z), scaled by batch size
        Tensor log_Y = OwnTensor::log(Y + 1e-9f); // Add epsilon for stability
        Tensor gY = (log_Y + 1.0f - log_softmax_z) * inv_batch_size;
        Y_node->grad += gy * gY;
    }
}

// ----- Other Math -----
// ===================================================================
// vjp_Div
// ===================================================================
void vjp_Div(Node* n, const Tensor& gy){
    Node* A_node = n->inputs[0].get();
    Node* B_node = n->inputs[1].get();
    const Tensor& A = A_node->value;
    const Tensor& B = B_node->value;

    if (A_node->requires_grad()) {
        A_node->grad += gy / B;
    }
    if (B_node->requires_grad()) {
        B_node->grad += (gy * -1.0f * A) / (B * B);
    }
}

// ===================================================================
// vjp_Linear
// ===================================================================
void vjp_Linear(Node* n, const Tensor& gy){
    Node* X_node = n->inputs[0].get(); // Input X
    Node* W_node = n->inputs[1].get(); // Weight W
    Node* b_node = n->inputs[2].get(); // Bias b
    
    const Tensor& X = X_node->value;
    const Tensor& W = W_node->value;

    // --- FIX START ---
     if (X_node->requires_grad()) {
        // [batch, out] @ [out, in] -> [batch, in] (Correct shape for dX)
        X_node->grad += OwnTensor::matmul(gy, W.t());
    }

    // VJP for weight W: dW = X.T @ dY
    if (W_node->requires_grad()) {
        // [in, batch] @ [batch, out] -> [in, out] (Correct shape for dW)
        W_node->grad += OwnTensor::matmul(X.t(), gy);
    }
    // --- FIX END ---

    // VJP for bias b is correct
    if (b_node->requires_grad()) {
        b_node->grad += OwnTensor::reduce_sum(gy, {0}, false);
    }
}
// ===================================================================
// vjp_Reciprocal
// ===================================================================
void vjp_Reciprocal(Node* n, const Tensor& gy){
    Node* X_node = n->inputs[0].get();
    if (!X_node->requires_grad()) return;
    const Tensor& X = X_node->value;
    // VJP is gy * (-1 / X^2)
    X_node->grad += gy * -1.0f / (X * X);
}

// ===================================================================
// vjp_RealRMSNorm (Stub)
// ===================================================================
void vjp_RealRMSNorm(Node* n, const Tensor& gy){
    throw std::runtime_error("VJP for RealRMSNorm not implemented yet!");
}

// ===================================================================
// vjp_Cosh
// ===================================================================
void vjp_Cosh(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad()) return;

    // VJP is gy * sinh(x)
    X->grad += gy * OwnTensor::sinh(X->value);
}

// ===================================================================
// vjp_Sinh
// ===================================================================
void vjp_Sinh(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad()) return;

    // VJP is gy * cosh(x)
    X->grad += gy * OwnTensor::cosh(X->value);
}

// ===================================================================
// vjp_Sign
// ===================================================================
void vjp_Sign(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad()) return;

    // The gradient is zero. We add gy * 0 to correctly handle shapes
    // in case of broadcasting.
    X->grad += gy * 0.0f;
}

// ===================================================================
// vjp_Cos
// ===================================================================
void vjp_Cos(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad()) return;

    // VJP is gy * -sin(x)
    X->grad += gy * -1.0f * OwnTensor::sin(X->value);
}

// ===================================================================
// vjp_Sin
// ===================================================================
void vjp_Sin(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad()) return;

    // VJP is gy * cos(x)
    X->grad += gy * OwnTensor::cos(X->value);
}


// =================================================================== 
//  vjp_Sqrt
// ===================================================================
void vjp_Sqrt(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad()) return;

    // VJP is gy * (0.5 / sqrt(x)) = gy * 0.5 / y
    // n->value is the result of the forward pass, which is sqrt(x).
    X->grad += gy * 0.5f / n->value;
}

// ===================================================================
// vjp_Relumask
// ===================================================================
void vjp_Relumask(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad()) return;

    // The gradient is 0. We multiply by gy to ensure correct broadcasting
    // for a zero-like tensor.
    X->grad += gy * 0.0f;
}

// ===================================================================
// vjp_RELUAtt
// ===================================================================
void vjp_RELUAtt(Node* n, const Tensor& gy){
    Node* A = n->inputs[0].get(), *B = n->inputs[1].get(), *C = n->inputs[2].get(), *D = n->inputs[3].get();
    const Tensor& q = *n->tape[0], &k = *n->tape[1], &v = *n->tape[2], &s = *n->tape[3];
    
    float scale = 1.0f / std::sqrt(static_cast<float>(k.shape().dims.back()));

    // VJP for the final matmul: y = s @ v
    Tensor dL_ds = matmul(gy, v.t());
    Tensor dL_dv = matmul(s.t(), gy);
    
    // VJP for the ReLU: s = relu(g). Gradient is dL/ds * (g > 0)
    // We get the original 'g' by inverting the relu on 's': where s is 0, g was <=0.
    // The mask is simply where s > 0.
    const float epsilon = 1e-9f;
    Tensor sign_s = s / (OwnTensor::abs(s, ag::current_stream()) + epsilon);
    Tensor relu_mask = (sign_s + OwnTensor::abs(sign_s, ag::current_stream())) * 0.5f; // relu(sign(s)) is 1 where s>0
    Tensor dL_dg = dL_ds * relu_mask;
    
    // VJP for the scaled matmul: g = (q @ k.T) * scale
    Tensor dL_dq = matmul(dL_dg, k) * scale;
    Tensor dL_dk = matmul(dL_dg.t(), q) * scale;

    // Propagate gradients to the weight matrices and the input A
    if (B->requires_grad()) B->grad += matmul(A->value.t(), dL_dq);
    if (C->requires_grad()) C->grad += matmul(A->value.t(), dL_dk);
    if (D->requires_grad()) D->grad += matmul(A->value.t(), dL_dv);
    if (A->requires_grad()) {
        A->grad += matmul(dL_dq, B->value) + 
                   matmul(dL_dk, C->value) + 
                   matmul(dL_dv, D->value);
    }
}

// ===================================================================
// vjp_MOE
// ===================================================================
void vjp_MOE(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    Node* W = n->inputs[1].get();
    Node* B = n->inputs[2].get();

    // VJP for input X: dX = dY @ W.T
    if (X->requires_grad()) {
        X->grad += OwnTensor::matmul(gy, W->value.t());
    }
    // VJP for weight W: dW = X.T @ dY
    if (W->requires_grad()) {
        W->grad += OwnTensor::matmul(X->value.t(), gy);
    }
    // VJP for bias B: dB is the sum of gradients along the batch dimension
    if (B->requires_grad()) {
        B->grad += OwnTensor::reduce_sum(gy, {0}, false);
    }
}
// ===================================================================
// vjp_SigAtt
// ===================================================================
void vjp_SigAtt(Node* n, const Tensor& gy){
    Node* A = n->inputs[0].get(), *B = n->inputs[1].get(), *C = n->inputs[2].get(), *D = n->inputs[3].get();
    const Tensor& q = *n->tape[0], &k = *n->tape[1], &v = *n->tape[2], &s = *n->tape[3];

    float scale = 1.0f / std::sqrt(static_cast<float>(k.shape().dims.back()));

    // VJP for the final matmul: y = s @ v
    Tensor dL_ds = matmul(gy, v.t());
    Tensor dL_dv = matmul(s.t(), gy);
    
    // VJP for the Sigmoid activation: s = sigmoid(g)
    // dL/dg = dL/ds * (s * (1 - s))
    Tensor dL_dg = dL_ds * (s * (1.0f - s));
    
    // VJP for the scaled matmul: g = (q @ k.T) * scale
    Tensor dL_dq = matmul(dL_dg, k) * scale;
    Tensor dL_dk = matmul(dL_dg.t(), q) * scale;

    // Propagate gradients to the weight matrices and the input A
    if (B->requires_grad()) B->grad += matmul(A->value.t(), dL_dq);
    if (C->requires_grad()) C->grad += matmul(A->value.t(), dL_dk);
    if (D->requires_grad()) D->grad += matmul(A->value.t(), dL_dv);
    if (A->requires_grad()) {
        A->grad += matmul(dL_dq, B->value) + 
                   matmul(dL_dk, C->value) + 
                   matmul(dL_dv, D->value);
    }
}

// ----- Loss Functions -----
// ===================================================================
// vjp_MSELoss
// ===================================================================

void vjp_MSELoss(Node* n, const Tensor& gy){
    Node* Z_node = n->inputs[0].get(); // Predictions
    Node* Y_node = n->inputs[1].get(); // Targets
    const Tensor& Z = Z_node->value;
    const Tensor& Y = Y_node->value;
    
    const float N = static_cast<float>(Z.numel());
    const float scale = 2.0f / N;

    Tensor diff = Z - Y;
    
    if (Z_node->requires_grad()) {
        // gy is a scalar from the loss. It will be broadcast.
        Z_node->grad += gy * diff * scale;
    }
    if (Y_node->requires_grad()) {
        Y_node->grad += gy * diff * (-1.0f * scale);
    }
}

// ===================================================================
// vjp_MAELoss
// ===================================================================
void vjp_MAELoss(Node* n, const Tensor& gy){
    Node* Z_node = n->inputs[0].get();
    Node* Y_node = n->inputs[1].get();
    const Tensor& Z = Z_node->value;
    const Tensor& Y = Y_node->value;

    const float inv_N = 1.0f / static_cast<float>(Z.numel());

    // Since OwnTensor doesn't have a sign op, we build it arithmetically
    const float epsilon = 1e-9f;
    Tensor diff = Z - Y;
    Tensor sign_diff = diff / (OwnTensor::abs(diff, ag::current_stream()) + epsilon);
    
    if (Z_node->requires_grad()) {
        Z_node->grad += gy * sign_diff * inv_N;
    }
    if (Y_node->requires_grad()) {
        Y_node->grad += gy * sign_diff * (-1.0f * inv_N);
    }
}

void vjp_Leaf(Node*, const Tensor&){ /* no-op */ }

} // namespace detail


// -------- dispatch table --------
VjpFn vjp_lookup(Op op){
    switch(op){
#define OP(name, arity, str) case Op::name: return &detail::vjp_##name;
#include "ad/detail/ops.def"
#undef OP
        default: return nullptr;
    }
}

} // namespace ag
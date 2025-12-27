// ====================================================================
// FILE: cgadimpl/src/autodiff/autodiff_vjp_ops.cpp (GPU-Aware Version)
// ====================================================================

#include "ad/detail/autodiff_ops.hpp"
#include "ad/runtime/runtime.hpp"
#include <cmath>
#include <stdexcept> // Required for std::runtime_error

namespace ag {
namespace detail{

static Tensor reduce_for_broadcast(const Tensor& grad_in, const Tensor& target_val) {
    // If shapes already match, no reduction is needed.
    if (grad_in.shape().dims == target_val.shape().dims) {
        return grad_in;
    }

    const auto& grad_dims = grad_in.shape().dims;
    const auto& target_dims = target_val.shape().dims;
    
    std::vector<int64_t> axes_to_sum;
    int grad_ndim = grad_dims.size();
    int target_ndim = target_dims.size();
    
    // Find which axes of the incoming gradient need to be summed.
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
    
    // Reshape to exactly match the target shape (e.g., from [1, 5] to [5]).
    return summed_grad.reshape(target_val.shape());
}

// // ----- elementwise binary -----
// // Correct: Accumulates gradient for both parents.
// // ----- elementwise binary -----
// // Correct: Accumulates gradient for both parents.
void vjp_Add(Node* n, const Tensor& gy){
    Node* A = n->next_edges[0].function.get(); 
    Node* B = n->next_edges[1].function.get();
    if (A->requires_grad())
    { 
        std::lock_guard<std::mutex> lock(A->mutex_);
        A->tensor.grad_view() += reduce_for_broadcast(gy, A->tensor);
    }
    if (B->requires_grad())
    { 
        std::lock_guard<std::mutex> lock(B->mutex_);
        B->tensor.grad_view() += reduce_for_broadcast(gy, B->tensor);
    }
}

void vjp_Sub(Node* n, const Tensor& gy){
    Node* A = n->next_edges[0].function.get();
    Node* B = n->next_edges[1].function.get();
    if (A->requires_grad()) 
    {
        std::lock_guard<std::mutex> lock(A->mutex_);
        A->tensor.grad_view() += reduce_for_broadcast(gy, A->tensor);}
    if (B->requires_grad()) 
    {
        std::lock_guard<std::mutex> lock(B->mutex_);
        B->tensor.grad_view() += reduce_for_broadcast(gy * -1.0f, B->tensor);}
}

void vjp_Mul(Node* n, const Tensor& gy){
    Node* A = n->next_edges[0].function.get();
    Node* B = n->next_edges[1].function.get();
    if (A->requires_grad()) 
    {
        std::lock_guard<std::mutex> lock(A->mutex_);
        A->tensor.grad_view() += reduce_for_broadcast(gy * B->tensor, A->tensor);}
    if (B->requires_grad()) 
    {
        std::lock_guard<std::mutex> lock(B->mutex_);
        B->tensor.grad_view() += reduce_for_broadcast(gy * A->tensor, B->tensor);}
}
void vjp_Div(Node* n, const Tensor& gy){
    Node* A = n->next_edges[0].function.get();
    Node* B = n->next_edges[1].function.get();

    // VJP for A: dL/dA = gy * (1/B)
    if (A->requires_grad()) 
    { 
        std::lock_guard<std::mutex> lock(A->mutex_);            
        A->tensor.grad_view() += reduce_for_broadcast(gy / B->tensor, A->tensor);
    }
    
    // VJP for B: dL/dB =    // VJP for Input 'x'
    if (B->requires_grad()) {
        std::lock_guard<std::mutex> lock(B->mutex_);
        Tensor grad_B = gy * -1.0f * A->tensor / (B->tensor * B->tensor);
        B->tensor.grad_view() += reduce_for_broadcast(grad_B, B->tensor);
    }
}

// ----- elementwise trinary & matmul -----
// ===================================================================
// vjp_FMA
// ===================================================================
void vjp_FMA(Node* n, const Tensor& gy){
    Node* A = n->next_edges[0].function.get();
    Node* B = n->next_edges[1].function.get();
    Node* C = n->next_edges[2].function.get();
    
    const Tensor& At = A->tensor;
    const Tensor& Bt = B->tensor;

    // The OwnTensor operators handle device, stream, and broadcasting automatically.
    if (A->requires_grad()){
        std::lock_guard<std::mutex> lock(A->mutex_);
        A->tensor.grad_view() += OwnTensor::matmul(gy, Bt.t());
    }
    if (B->requires_grad()){
        std::lock_guard<std::mutex> lock(B->mutex_);
        B->tensor.grad_view() += OwnTensor::matmul(At.t(), gy);
    }
    if (C->requires_grad()) {
        std::lock_guard<std::mutex> lock(C->mutex_);
        C->tensor.grad_view() += gy;
    }
}


// ===================================================================
// vjp_LayerNorm
// ===================================================================
void vjp_LayerNorm(Node* n, const Tensor& gy){
    Node* x = n->next_edges[0].function.get();
    if (!x->requires_grad()) return;

    // Get the number of features from the last dimension
    const float N = static_cast<float>(x->tensor.shape().dims.back());
    
    const Tensor& variance = *n->tape[0];
    const Tensor& mean = *n->tape[1];

    Tensor std_dev = OwnTensor::sqrt(variance + 1e-5f, ag::current_stream()); // Use OwnTensor::sqrt
    Tensor xmu = x->tensor - mean;

    // Use reduce_sum for row-wise summation
    Tensor grad_sum = OwnTensor::reduce_sum(gy, {-1}, true);
    Tensor grad_dot_xmu = OwnTensor::reduce_sum(gy * xmu, {-1}, true);

    Tensor term1 = gy * N;
    Tensor term2 = term1 - grad_sum;
    Tensor term3 = term2 - (xmu * (grad_dot_xmu / (variance + 1e-5f)));
    Tensor dx = term3 / (std_dev * N);
    if (x->requires_grad()){
        std::lock_guard<std::mutex> lock(x->mutex_);
        x->tensor.grad_view() += dx;
    }
}

// ===================================================================
// vjp_RMSNorm
// ===================================================================
void vjp_RMSNorm(Node* n, const Tensor& gy){
    Node* x = n->next_edges[0].function.get();
    if (!x->requires_grad()) return;

    const Tensor& rms = *n->tape[0]; // rsqrt(variance + epsilon)
    const Tensor& y_normalized = *n->tape[1]; // x * rsqrt(...)
    
    const float inv_N = 1.0f / static_cast<float>(x->tensor.shape().dims.back());

    Tensor dot = OwnTensor::reduce_sum(gy * y_normalized, {-1}, true);
    
    // grad_x = rsqrt * (gy - y * dot)
    Tensor grad_x = rms * (gy - y_normalized * dot);

    if (x->requires_grad()){
        std::lock_guard<std::mutex> lock(x->mutex_);
        x->tensor.grad_view() += grad_x;
    }
}

// ===================================================================
// vjp_RealLayerNorm
// ===================================================================
void vjp_RealLayerNorm(Node* n, const Tensor& gy){
    Node* x = n->next_edges[0].function.get();
    Node* g = n->next_edges[1].function.get(); // Gain
    Node* b = n->next_edges[2].function.get(); // Bias
    
    const float N = static_cast<float>(x->tensor.shape().dims.back());

    const Tensor& variance = *n->tape[0];
    const Tensor& mean = *n->tape[1];
    const Tensor& x_normalized = *n->tape[2];
    
    Tensor std_dev = OwnTensor::sqrt(variance + 1e-5f, ag::current_stream());
    Tensor xmu = x->tensor - mean;

    // VJP for the core normalization part (same as LayerNorm)
    Tensor grad_sum = OwnTensor::reduce_sum(gy, {-1}, true);
    Tensor grad_dot_xmu = OwnTensor::reduce_sum(gy * x_normalized, {-1}, true);

    Tensor term1 = gy * N;
    Tensor term2 = term1 - grad_sum;
    Tensor term3 = term2 - (x_normalized * grad_dot_xmu);
    Tensor dx_normalized = term3 / N;

    // VJP for the affine transformation (scale and shift)
    if (g->requires_grad()) {
        std::lock_guard<std::mutex> lock(g->mutex_);
        g->tensor.grad_view() += OwnTensor::reduce_sum(gy * x_normalized, {-1});
    }
    if (b->requires_grad()) {
        std::lock_guard<std::mutex> lock(b->mutex_);
        b->tensor.grad_view() += OwnTensor::reduce_sum(gy, {-1});
    }
    if (x->requires_grad()) {
        std::lock_guard<std::mutex> lock(x->mutex_);
        x->tensor.grad_view() += (g->tensor / std_dev) * dx_normalized;
    }
}

// ----- Attention Mechanisms -----
// ===================================================================
// vjp_Attention
// ===================================================================
void vjp_Attention(Node* n, const Tensor& gy){
    Node* A = n->next_edges[0].function.get();
    Node* B = n->next_edges[1].function.get();
    Node* C = n->next_edges[2].function.get();
    Node* D = n->next_edges[3].function.get();
    
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
        std::lock_guard<std::mutex> lock(B->mutex_);
        B->tensor.grad_view() += OwnTensor::matmul(A->tensor.t(), dL_dq) * scale;
    }
    if (C->requires_grad()) {
        std::lock_guard<std::mutex> lock(C->mutex_);
        C->tensor.grad_view() += OwnTensor::matmul(A->tensor.t(), dL_dk) * scale;
    }
    if (D->requires_grad()) {
        std::lock_guard<std::mutex> lock(D->mutex_);
        D->tensor.grad_view() += OwnTensor::matmul(A->tensor.t(), dL_dv);
    }
    if (A->requires_grad()) {
        std::lock_guard<std::mutex> lock(A->mutex_);
        Tensor dL_dA_q = OwnTensor::matmul(dL_dq, B->tensor);
        Tensor dL_dA_k = OwnTensor::matmul(dL_dk, C->tensor);
        Tensor dL_dA_v = OwnTensor::matmul(dL_dv, D->tensor);
        A->tensor.grad_view() += (dL_dA_q * scale) + (dL_dA_k * scale) + dL_dA_v;
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
    Node* X = n->next_edges[0].function.get();
    Node* A = n->next_edges[1].function.get();
    Node* B = n->next_edges[2].function.get();
    Node* C = n->next_edges[3].function.get();
    Node* D = n->next_edges[4].function.get();

    // Recompute intermediates using OwnTensor API
    Tensor y = OwnTensor::matmul(X->tensor, A->tensor.t()) + B->tensor;
    Tensor h = OwnTensor::matmul(X->tensor, C->tensor.t()) + D->tensor;

    // --- Re-implement sigmoid and its derivative using OwnTensor ops ---
    Tensor sig_y = 1.0f / (1.0f + OwnTensor::exp(y * -1.0f));
    Tensor swish_y = y * sig_y;
    Tensor swish_grad = sig_y + y * sig_y * (1.0f - sig_y);
    // ---

    // Propagate gradients
    Tensor dL_dh = swish_y * gy;
    Tensor dL_dy = h * swish_grad * gy;

    if (D->requires_grad()) {
        std::lock_guard<std::mutex> lock(D->mutex_);
        D->tensor.grad_view() += dL_dh;
    }
    if (C->requires_grad()) {
        std::lock_guard<std::mutex> lock(C->mutex_);
        C->tensor.grad_view() += OwnTensor::matmul(dL_dh.t(), X->tensor);
    }
    
    if (B->requires_grad()){
        std::lock_guard<std::mutex> lock(B->mutex_);
        B->tensor.grad_view() += dL_dy;
    }
    if (A->requires_grad()) {
        std::lock_guard<std::mutex> lock(A->mutex_);
        A->tensor.grad_view() += OwnTensor::matmul(dL_dy.t(), X->tensor);
    }
    
    if (X->requires_grad()) {
        std::lock_guard<std::mutex> lock(X->mutex_);
        X->tensor.grad_view() += OwnTensor::matmul(dL_dh, C->tensor) + OwnTensor::matmul(dL_dy, A->tensor);
    }
}
// ===================================================================
// vjp_Relu
// ===================================================================
void vjp_Relu(Node* n, const Tensor& gy){
    Node* input_node = n->next_edges[0].function.get();
    if(input_node->requires_grad()){
        std::lock_guard<std::mutex> lock(input_node->mutex_);
        
        cudaStream_t stream = (cudaStream_t)ag::current_stream();
        const float epsilon = 1e-9f;
        
        Tensor input_tensor = input_node->tensor;
        Tensor sign_X = input_tensor / (OwnTensor::abs(input_tensor, stream) + epsilon);
        Tensor mask = (sign_X + OwnTensor::abs(sign_X, stream)) * 0.5f;

        input_node->tensor.grad_view() += (gy * mask);
    }
}
// ===================================================================
// vjp_Exp
// ===================================================================

void vjp_Exp(Node* n, const Tensor& gy){
    Node* X = n->next_edges[0].function.get();
    if (!X->requires_grad()) return;

    // The VJP for exp(x) is gy * exp(x). The forward pass output is exp(x).
    // This uses the stream-aware OwnTensor operator '*' for both CPU and GPU.
    if (X->requires_grad()){
        std::lock_guard<std::mutex> lock(X->mutex_);
        X->tensor.grad_view() += gy * n->tensor;
    }
}

// ===================================================================
// vjp_Log
// ===================================================================

void vjp_Log(Node* n, const Tensor& gy){
    Node* X = n->next_edges[0].function.get();
    if (!X->requires_grad()) return;

    // The VJP for log(x) is gy / x.
    // This uses the stream-aware OwnTensor operator '/' for both CPU and GPU.
    if (X->requires_grad()){
        std::lock_guard<std::mutex> lock(X->mutex_);
        X->tensor.grad_view() += gy / X->tensor;
    }
}


// ===================================================================
// vjp_GCU
// ===================================================================
void vjp_GCU(Node* n, const Tensor& gy){
    Node* X = n->next_edges[0].function.get();
    if (!X->requires_grad()) return;

    // VJP is gy * (cos(x) - x * sin(x))
    // All ops are from OwnTensor and are stream-aware.
    Tensor d_gcu = OwnTensor::cos(X->tensor) - (X->tensor * OwnTensor::sin(X->tensor));
    if (X->requires_grad()){
        std::lock_guard<std::mutex> lock(X->mutex_);
        X->tensor.grad_view()+= gy * d_gcu;
    }
}

// ===================================================================
// vjp_Mish
// ===================================================================
void vjp_Mish(Node* n, const Tensor& gy){
    Node* X = n->next_edges[0].function.get();
    if (!X->requires_grad()) return;

    // Re-calculate intermediates needed for the derivative
    // softplus(x) = log(1 + exp(x))
    Tensor sp = OwnTensor::log(1.0f + OwnTensor::exp(X->tensor));
    Tensor tanh_sp = OwnTensor::tanh(sp);
    
    // sigmoid(x) = 1 / (1 + exp(-x))
    Tensor sig_x = 1.0f / (1.0f + OwnTensor::exp(X->tensor * -1.0f));

    // The derivative of mish
    Tensor d_mish = tanh_sp + X->tensor * (1.0f - (tanh_sp * tanh_sp)) * sig_x;
    
    // Apply the chain rule
    if (X->requires_grad()){
        std::lock_guard<std::mutex> lock(X->mutex_);
        X->tensor.grad_view() += gy * d_mish;
    }
}

// ===================================================================
// vjp_Tanh
// ===================================================================
void vjp_Tanh(Node* n, const Tensor& gy){
    Node* X = n->next_edges[0].function.get();
    if (!X->requires_grad()) return;

    // VJP is gy * (1 - tanh(x)^2)
    // Here, t = n->tensor is the result of the forward tanh(x)
    const Tensor& t = n->tensor;
    if (X->requires_grad()){
        std::lock_guard<std::mutex> lock(X->mutex_);
        X->tensor.grad_view() += gy * (1.0f - (t * t));
    }
}

// ===================================================================
// vjp_Sigmoid
// ===================================================================
void vjp_Sigmoid(Node* n, const Tensor& gy){
    Node* X = n->next_edges[0].function.get();
    if (!X->requires_grad()) return;

    // VJP is gy * (sigmoid(x) * (1 - sigmoid(x)))
    // Here, s = n->tensor is the result of the forward sigmoid(x)
    const Tensor& s = n->tensor;
    if (X->requires_grad()){
        std::lock_guard<std::mutex> lock(X->mutex_);
        X->tensor.grad_view() += gy * (s * (1.0f - s));
    }
}


// ===================================================================
// vjp_Softplus
// ===================================================================
void vjp_Softplus(Node* n, const Tensor& gy){
    Node* X = n->next_edges[0].function.get();
    if (!X->requires_grad()) return;

    // VJP is gy * sigmoid(x)
    // sigmoid(x) = 1 / (1 + exp(-x))
    Tensor d_softplus = 1.0f / (1.0f + OwnTensor::exp(X->tensor * -1.0f));
    
    if (X->requires_grad()){
        std::lock_guard<std::mutex> lock(X->mutex_);
        X->tensor.grad_view()  += gy * d_softplus;
    }
}


// ===================================================================
// vjp_Gaus
// ===================================================================
void vjp_Gaus(Node* n, const Tensor& gy){
    Node* X = n->next_edges[0].function.get();
    if (!X->requires_grad()) return;

    // VJP is gy * (-2 * x * exp(-x^2))
    // We can reuse the forward pass output, n->tensor, which is exp(-x^2).
    if (X->requires_grad()) {
        std::lock_guard<std::mutex> lock(X->mutex_);
        X->tensor.grad_view() += gy * -2.0f * X->tensor * n->tensor;
    }
}

// ===================================================================
// vjp_Transpose
// ===================================================================
void vjp_Transpose(Node* n, const Tensor& gy){
    // Transpose is a single-input operation
    Node* input_node = n->next_edges[0].function.get();
    
    // Safety check
    if (!input_node) return;

    if (input_node->requires_grad()) {
        std::lock_guard<std::mutex> lock(input_node->mutex_);
        
        // The gradient of transpose(X) back to X is transpose(G)
        // We use the same .t() operation on the incoming gradient
        Tensor grad_input = gy.t();
        
        input_node->tensor.grad_view() += grad_input;
    }
}

// ===================================================================
// vjp_SiLU
// ===================================================================
void vjp_SiLU(Node* n, const Tensor& gy){
    Node* X = n->next_edges[0].function.get();
    if (!X->requires_grad()) return;

    // Recompute sigmoid(x)
    Tensor s = 1.0f / (1.0f + OwnTensor::exp(X->tensor * -1.0f));
    
    // Derivative of SiLU
    Tensor d_silu = s * (1.0f + X->tensor * (1.0f - s));
    
    if (X->requires_grad()){
        std::lock_guard<std::mutex> lock(X->mutex_);
        X->tensor.grad_view() += gy * d_silu;}
}

// ===================================================================
// vjp_Parcon
// ===================================================================
void vjp_Parcon(Node* n, const Tensor& gy){
    Node* X = n->next_edges[0].function.get();
    if (!X->requires_grad()) return;

    // VJP is gy * (2 - 2*x)
    if (X->requires_grad()){
        std::lock_guard<std::mutex> lock(X->mutex_);
        X->tensor.grad_view() += gy * (2.0f - 2.0f * X->tensor);
    }
}

// ===================================================================
// vjp_LiSHT
// ===================================================================
void vjp_LiSHT(Node* n, const Tensor& gy){
    Node* X = n->next_edges[0].function.get();
    if (!X->requires_grad()) return;

    // Recompute tanh(x)
    Tensor th_x = OwnTensor::tanh(X->tensor);
    
    // Derivative is tanh(x) + x * sech(x)^2, which is tanh(x) + x * (1 - tanh(x)^2)
    Tensor d_lisht = th_x + X->tensor * (1.0f - (th_x * th_x));
    
    if (X->requires_grad()){
        std::lock_guard<std::mutex> lock(X->mutex_);
        X->tensor.grad_view() += gy * d_lisht;}
} 

// ===================================================================
// vjp_GELU
// ===================================================================
void vjp_GELU(Node* n, const Tensor& gy){
    Node* X_node = n->next_edges[0].function.get();
    if (!X_node->requires_grad()) return;
    const Tensor& x = X_node->tensor;

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
    if (X_node->requires_grad()){
        std::lock_guard<std::mutex> lock(X_node->mutex_);
        X_node->tensor.grad_view() += gy * d_gelu;
    }
}
// ===================================================================
// vjp_LeakyRelu
// ===================================================================
void vjp_LeakyRelu(Node* n, const Tensor& gy){
    Node* X_node = n->next_edges[0].function.get();
    if (!X_node->requires_grad()) return;
    const Tensor& x = X_node->tensor;
    
    // Get alpha from the second input node
    Node* A_node = n->next_edges[1].function.get();
    // Use .data<T>()[0] to get the scalar value from the 1x1 tensor
    // FIX: Move to CPU before data access to avoid "scalar copy: invalid argument" on CUDA
    float alpha = A_node->tensor.to_cpu().data<float>()[0]; 

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
    if (X_node->requires_grad()){
        std::lock_guard<std::mutex> lock(X_node->mutex_);
        X_node->tensor.grad_view() += gy * d_leaky;
    }
}

// ===================================================================
// vjp_MatMul
// ===================================================================
void vjp_MatMul(Node* n, const Tensor& gy){
    Node* A_node = n->next_edges[0].function.get();
    Node* B_node = n->next_edges[1].function.get();
    const Tensor& A = A_node->tensor;
    const Tensor& B = B_node->tensor;

    // VJP for A: dL/dA = dL/dY @ B^T
    if (A_node->requires_grad()) {
        std::lock_guard<std::mutex> lock(A_node->mutex_);
        A_node->tensor.grad_view() += OwnTensor::matmul(gy, B.t());
    }

    // VJP for B: dL/dB = A^T @ dL/dY
    if (B_node->requires_grad()) {
        std::lock_guard<std::mutex> lock(B_node->mutex_);
        B_node->tensor.grad_view() += OwnTensor::matmul(A.t(), gy);
    }
}

// ===================================================================
// vjp_Dyntanh
// ===================================================================
void vjp_Dyntanh(Node* n, const Tensor& gy){
    Node* X = n->next_edges[0].function.get(); 
    Node* A = n->next_edges[1].function.get(); 
    Node* B = n->next_edges[2].function.get(); 
    Node* G = n->next_edges[3].function.get();
    
    // The tape stores h = a*x from the forward pass.
    const Tensor& h = *(n->tape.back());
    Tensor th_h = OwnTensor::tanh(h);
    
    // Derivative of tanh(h) is 1 - tanh(h)^2
    Tensor d_tanh = 1.0f - (th_h * th_h);
    
    if (G->requires_grad()) {
        std::lock_guard<std::mutex> lock(G->mutex_);
        G->tensor.grad_view() += gy * th_h;
    }
    if (B->requires_grad()) {
        std::lock_guard<std::mutex> lock(B->mutex_);
        B->tensor.grad_view() += gy;
    }
    if (A->requires_grad()) {
        std::lock_guard<std::mutex> lock(A->mutex_);        
        // Chain rule: gy * g * d_tanh * x
        A->tensor.grad_view() += gy * G->tensor * d_tanh * X->tensor;
    }
    if (X->requires_grad()) {
        std::lock_guard<std::mutex> lock(X->mutex_);
        // Chain rule: gy * g * d_tanh * a
        X->tensor.grad_view() += gy * G->tensor * d_tanh * A->tensor;
    }
}

// ----- Reductions -----
// ===================================================================
// vjp_Sum
// ===================================================================
void vjp_Sum(Node* n, const Tensor& gy){
    Node* X = n->next_edges[0].function.get();
    if (!X->requires_grad()) return;

    // `gy` is a 1x1 scalar tensor. The '+' operator will automatically
    // broadcast it to the shape of X->tensor.grad_view().
    if(X->requires_grad()) {
        std::lock_guard<std::mutex> lock(X->mutex_);
        X->tensor.grad_view() += gy;
    }
}

// ===================================================================
// vjp_RowSum
// ===================================================================
void vjp_RowSum(Node* n, const Tensor& gy){
    Node* X = n->next_edges[0].function.get();
    if (!X->requires_grad()) return;

    // `gy` has shape [B, 1]. The '+' operator will automatically
    // broadcast it to the shape of X->tensor.grad_view(), which is [B, C].
    if (X->requires_grad()) {
        std::lock_guard<std::mutex> lock(X->mutex_);
        X->tensor.grad_view() += gy;
    }
}

// ===================================================================
// vjp_RowMax
// ===================================================================
void vjp_RowMax(Node* n, const Tensor& gy){
    // To implement this, we need to know the *index* of the max element in each row.
    // This requires an `argmax` function which is not available in OwnTensor.
    // The VJP would look like:
    //   Tensor indices = OwnTensor::argmax(X->tensor, /*axis=*/-1);
    //   X->tensor.grad_view().scatter_add_(/*axis=*/-1, indices, gy);
    throw std::runtime_error("VJP for RowMax cannot be implemented without argmax or comparison ops in the tensor library.");
}

// ===================================================================
// vjp_MeanAll
// ===================================================================

void vjp_MeanAll(Node* n, const Tensor& gy){
    Node* X = n->next_edges[0].function.get();
    if (!X->requires_grad()) return;
    
    // The gradient needs to be scaled by 1/N.
    float scale = 1.0f / static_cast<float>(X->tensor.numel());
    
    // `gy` is a scalar. `gy * scale` is also a scalar.
    // The `+=` operator will broadcast this scalar across the entire gradient tensor.
    if(X->requires_grad()){
        std::lock_guard<std::mutex> lock(X->mutex_);
        X->tensor.grad_view() += gy * scale;}
}

// ===================================================================
// vjp_SoftmaxRow
// ===================================================================
void vjp_SoftmaxRow(Node* n, const Tensor& gy){
    Node* Z = n->next_edges[0].function.get();
    if (!Z->requires_grad()) return;

    // y is the output of the softmax, which is stored in the node's value.
    const Tensor& y = n->tensor;

    // Calculate the dot product along the rows.
    // This needs to be a sum, not a matmul.
    Tensor dot = OwnTensor::reduce_sum(y * gy, {-1}, true);

    // The += operator will broadcast 'dot' correctly.
    if (Z->requires_grad()){
        std::lock_guard<std::mutex> lock(Z->mutex_);
        Z->tensor.grad_view() += y * (gy - dot);
    }
}

// ===================================================================
// vjp_LogSumExpRow
// ===================================================================
void vjp_LogSumExpRow(Node* n, const Tensor& gy){
    Node* Z = n->next_edges[0].function.get();
    if (!Z->requires_grad()) return;

    // --- Re-implement softmax using OwnTensor ops ---
    // This is the derivative of logsumexp.
    const Tensor& z_val = Z->tensor;
    Tensor max_val = OwnTensor::reduce_max(z_val, {-1}, true);
    Tensor exp_z = OwnTensor::exp(z_val - max_val);
    Tensor sum_exp_z = OwnTensor::reduce_sum(exp_z, {-1}, true);
    Tensor softmax_z = exp_z / sum_exp_z;
    // ---
    
    // The VJP is gy * softmax(z). The += operator will handle broadcasting.
    if(Z->requires_grad()){
        std::lock_guard<std::mutex> lock(Z->mutex_);
        Z->tensor.grad_view() += gy * softmax_z;}
}

// ===================================================================
// vjp_CeWithLogits
// ===================================================================
void vjp_CeWithLogits(Node* n, const Tensor& gy){
    Node* Z_node = n->next_edges[0].function.get();
    Node* Y_node = n->next_edges[1].function.get();
    const Tensor& Z = Z_node->tensor;
    const Tensor& Y = Y_node->tensor;
    
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
        std::lock_guard<std::mutex> lock(Z_node->mutex_);
        // gZ = (softmax(Z) - Y) / batch_size
        Tensor gZ = (softmax_z - Y) * inv_batch_size;
        // The `gy` for a loss function is typically a scalar. The operators will broadcast it.
        Z_node->tensor.grad_view() += gy * gZ;
    }
    if (Y_node->requires_grad()) {
        std::lock_guard<std::mutex> lock(Y_node->mutex_);
        // gY = -log_softmax(Z) / batch_size
        Tensor gY = log_softmax_z * (-1.0f * inv_batch_size);
        Y_node->tensor.grad_view() += gy * gY;
    }
}

// ===================================================================
// vjp_KLDivergence
// ===================================================================
void vjp_KLDivergence(Node* n, const Tensor& gy){
    Node* Z_node = n->next_edges[0].function.get();
    Node* Y_node = n->next_edges[1].function.get();
    const Tensor& Z = Z_node->tensor;
    const Tensor& Y = Y_node->tensor;
    
    const float inv_batch_size = 1.0f / static_cast<float>(Z.shape().dims[0]);

    // Re-calculate stable softmax and log_softmax
    Tensor max_val = OwnTensor::reduce_max(Z, {-1}, true);
    Tensor z_shifted = Z - max_val;
    Tensor exp_z = OwnTensor::exp(z_shifted);
    Tensor sum_exp_z = OwnTensor::reduce_sum(exp_z, {-1}, true);
    Tensor softmax_z = exp_z / sum_exp_z;
    Tensor log_softmax_z = z_shifted - OwnTensor::log(sum_exp_z);

    if (Z_node->requires_grad()) {
        std::lock_guard<std::mutex> lock(Z_node->mutex_);
        // gZ = softmax(Z) - Y, scaled by batch size
        Tensor gZ = (softmax_z - Y) * inv_batch_size;
        Z_node->tensor.grad_view() += gy * gZ;
    }
    if (Y_node->requires_grad()) {
        std::lock_guard<std::mutex> lock(Y_node->mutex_);
        // gY = log(Y) + 1 - log_softmax(Z), scaled by batch size
        Tensor log_Y = OwnTensor::log(Y + 1e-9f); // Add epsilon for stability
        Tensor gY = (log_Y + 1.0f - log_softmax_z) * inv_batch_size;
        Y_node->tensor.grad_view() += gy * gY;
    }
}


// ===================================================================
// vjp_Linear
// ===================================================================
void vjp_Linear(Node* n, const Tensor& gy){
    Node* X_node = n->next_edges[0].function.get(); // Input X
    Node* W_node = n->next_edges[1].function.get(); // Weight W
    Node* b_node = n->next_edges[2].function.get(); // Bias b
    
    const Tensor& X = X_node->tensor;
    const Tensor& W = W_node->tensor;

    // VJP for input X: dX = dY @ W. Correct.
    // [B, Out] @ [Out, In] -> [B, In]
    if (X_node->requires_grad()) {
        std::lock_guard<std::mutex> lock(X_node->mutex_);
        X_node->tensor.grad_view() += OwnTensor::matmul(gy, W);
    }

    // VJP for weight W: dW = dY.T @ X. Correct math for Y = X @ W.T + b
    // [Out, B] @ [B, In] -> [Out, In]
    if (W_node->requires_grad()) {
        std::lock_guard<std::mutex> lock(W_node->mutex_);
        W_node->tensor.grad_view() += OwnTensor::matmul(gy.t(), X);
    }

    // VJP for bias b: sum(dY) over batch dimension, keeping rank. Correct.
    // [B, Out] -> reduce(axis=0) -> [1, Out]
    if (b_node->requires_grad()) {
        std::lock_guard<std::mutex> lock(b_node->mutex_);
        // Change keepdim from 'false' to 'true'.
        // This makes the result [1, Out] instead of [Out].
        b_node->tensor.grad_view() += OwnTensor::reduce_sum(gy, {0}, true);
    }
}
// ===================================================================
// vjp_Reciprocal
// ===================================================================
void vjp_Reciprocal(Node* n, const Tensor& gy){
    Node* X_node = n->next_edges[0].function.get();
    if (!X_node->requires_grad()) return;
    const Tensor& X = X_node->tensor;
    // VJP is gy * (-1 / X^2)
    if(X_node->requires_grad()){
        std::lock_guard<std::mutex> lock(X_node->mutex_);
        X_node->tensor.grad_view() += gy * -1.0f / (X * X);
    }
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
    Node* X = n->next_edges[0].function.get();
    if (!X->requires_grad()) return;

    // VJP is gy * sinh(x)
    if(X->requires_grad()){
        std::lock_guard<std::mutex> lock(X->mutex_);
        X->tensor.grad_view() += gy * OwnTensor::sinh(X->tensor);}
}

// ===================================================================
// vjp_Sinh
// ===================================================================
void vjp_Sinh(Node* n, const Tensor& gy){
    Node* X = n->next_edges[0].function.get();
    if (!X->requires_grad()) return;

    // VJP is gy * cosh(x)
    if(X->requires_grad()){
        std::lock_guard<std::mutex> lock(X->mutex_);
        X->tensor.grad_view() += gy * OwnTensor::cosh(X->tensor);
    }
}

// ===================================================================
// vjp_Sign
// ===================================================================
void vjp_Sign(Node* n, const Tensor& gy){
    Node* X = n->next_edges[0].function.get();
    if (!X->requires_grad()) return;

    // The gradient is zero. We add gy * 0 to correctly handle shapes
    // in case of broadcasting.
    if(X->requires_grad()){
        std::lock_guard<std::mutex> lock(X->mutex_);
        X->tensor.grad_view() += gy * 0.0f;
    }
}

// ===================================================================
// vjp_Cos
// ===================================================================
void vjp_Cos(Node* n, const Tensor& gy){
    Node* X = n->next_edges[0].function.get();
    if (!X->requires_grad()) return;

    // VJP is gy * -sin(x)
    if(X->requires_grad()){
        std::lock_guard<std::mutex> lock(X->mutex_);
        X->tensor.grad_view() += gy * -1.0f * OwnTensor::sin(X->tensor);
    }
}

// ===================================================================
// vjp_Sin
// ===================================================================
void vjp_Sin(Node* n, const Tensor& gy){
    Node* X = n->next_edges[0].function.get();
    if (!X->requires_grad()) return;

    // VJP is gy * cos(x)
    if(X->requires_grad()){
        std::lock_guard<std::mutex> lock(X->mutex_);
        X->tensor.grad_view() += gy * OwnTensor::cos(X->tensor);
    }
}
// ===================================================================
// vjp_Tan
// ===================================================================
void vjp_Tan(Node* n, const Tensor& gy){
    Node* X = n->next_edges[0].function.get();
    if (!X->requires_grad()) return;

    // VJP is gy * (1/cos(x)*1/cos(x))
    if (X->requires_grad()){
        std::lock_guard<std::mutex> lock(X->mutex_);
        X->tensor.grad_view() += gy * ((1/OwnTensor::cos(X->tensor)) * (1/OwnTensor::cos(X->tensor)));
    }
}

// ===================================================================
// vjp_Asin
// ===================================================================
void vjp_Asin(Node* n, const Tensor& gy){
    Node* X = n->next_edges[0].function.get();
    if (!X->requires_grad()) return;

    // VJP is gy * (1 / sqrt(1 - x^2))
    if(X->requires_grad()){
        std::lock_guard<std::mutex> lock(X->mutex_);
        X->tensor.grad_view() += gy * (1.0f / OwnTensor::sqrt(1.0f - (X->tensor * X->tensor), ag::current_stream()));
    }
}

// ===================================================================
// vjp_Acos
// ===================================================================
void vjp_Acos(Node* n, const Tensor& gy){
    Node* X = n->next_edges[0].function.get();
    if (!X->requires_grad()) return;

    // VJP is gy * (-1 / sqrt(1 - x^2))
    if(X->requires_grad()){
        std::lock_guard<std::mutex> lock(X->mutex_);
        X->tensor.grad_view() += gy * (-1.0f / OwnTensor::sqrt(1.0f - (X->tensor * X->tensor), ag::current_stream()));
    }
}

// ===================================================================
// vjp_Atan
// ===================================================================
void vjp_Atan(Node* n, const Tensor& gy){
    Node* X = n->next_edges[0].function.get();
    if (!X->requires_grad()) return;

    // VJP is gy * (1 / (1 + x^2))
    if(X->requires_grad()){
        std::lock_guard<std::mutex> lock(X->mutex_);
        X->tensor.grad_view() += gy * (1.0f / (1.0f + (X->tensor * X->tensor)));}
}


// =================================================================== 
//  vjp_Sqrt
// ===================================================================
void vjp_Sqrt(Node* n, const Tensor& gy){
    Node* X = n->next_edges[0].function.get();
    if (!X->requires_grad()) return;

    // VJP is gy * (0.5 / sqrt(x)) = gy * 0.5 / y
    // n->tensor is the result of the forward pass, which is sqrt(x).
    if (X->requires_grad()){
        std::lock_guard<std::mutex> lock(X->mutex_);
        X->tensor.grad_view() += gy * 0.5f / n->tensor;
    }
}

// ===================================================================
// vjp_Relumask
// ===================================================================
void vjp_Relumask(Node* n, const Tensor& gy){
    Node* X = n->next_edges[0].function.get();
    if (!X->requires_grad()) return;

    // The gradient is 0. We multiply by gy to ensure correct broadcasting
    // for a zero-like tensor.
    if(X->requires_grad()){
        std::lock_guard<std::mutex> lock(X->mutex_);
        X->tensor.grad_view() += gy * 0.0f;
    }
}

// ===================================================================
// vjp_RELUAtt
// ===================================================================
void vjp_RELUAtt(Node* n, const Tensor& gy){
    Node* A = n->next_edges[0].function.get(), *B = n->next_edges[1].function.get(), *C = n->next_edges[2].function.get(), *D = n->next_edges[3].function.get();
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
    if (D->requires_grad()) {
        std::lock_guard<std::mutex> lock(D->mutex_);
        D->tensor.grad_view() += matmul(A->tensor.t(), dL_dv);}
    if (B->requires_grad()) {
        std::lock_guard<std::mutex> lock(B->mutex_);
        B->tensor.grad_view() += matmul(A->tensor.t(), dL_dq);}
    if (C->requires_grad()) {
        std::lock_guard<std::mutex> lock(C->mutex_);
        C->tensor.grad_view() += matmul(A->tensor.t(), dL_dk);}
    if (A->requires_grad()) {
        std::lock_guard<std::mutex> lock(A->mutex_);
        A->tensor.grad_view() += matmul(dL_dq, B->tensor) + 
                   matmul(dL_dk, C->tensor) + 
                   matmul(dL_dv, D->tensor);
    }
}

// ===================================================================
// vjp_MOE
// ===================================================================
void vjp_MOE(Node* n, const Tensor& gy){
    Node* X = n->next_edges[0].function.get();
    Node* W = n->next_edges[1].function.get();
    Node* B = n->next_edges[2].function.get();

    // VJP for input X: dX = dY @ W.T
    if (X->requires_grad()) {
        std::lock_guard<std::mutex> lock(X->mutex_);
        X->tensor.grad_view() += OwnTensor::matmul(gy, W->tensor.t());
    }
    // VJP for weight W: dW = X.T @ dY
    if (W->requires_grad()) {
        std::lock_guard<std::mutex> lock(W->mutex_);
        W->tensor.grad_view() += OwnTensor::matmul(X->tensor.t(), gy);
    }
    // VJP for bias B: dB is the sum of gradients along the batch dimension
    if (B->requires_grad()) {
        std::lock_guard<std::mutex> lock(B->mutex_);
        B->tensor.grad_view() += OwnTensor::reduce_sum(gy, {0}, false);
    }
}
// ===================================================================
// vjp_SigAtt
// ===================================================================
void vjp_SigAtt(Node* n, const Tensor& gy){
    Node* A = n->next_edges[0].function.get(), *B = n->next_edges[1].function.get(), *C = n->next_edges[2].function.get(), *D = n->next_edges[3].function.get();
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
    if (B->requires_grad()) {
        std::lock_guard<std::mutex> lock(B->mutex_);
        
        B->tensor.grad_view() += matmul(A->tensor.t(), dL_dq);}
    if (C->requires_grad()) {
        std::lock_guard<std::mutex> lock(C->mutex_);
        
        C->tensor.grad_view() += matmul(A->tensor.t(), dL_dk);}
    if (D->requires_grad()) {
        std::lock_guard<std::mutex> lock(D->mutex_);
        
        D->tensor.grad_view() += matmul(A->tensor.t(), dL_dv);}
    if (A->requires_grad()) {
        std::lock_guard<std::mutex> lock(A->mutex_);
        A->tensor.grad_view() += matmul(dL_dq, B->tensor) + 
                   matmul(dL_dk, C->tensor) + 
                   matmul(dL_dv, D->tensor);
    }
}

// ----- Loss Functions -----
// ===================================================================
// vjp_MSELoss
// ===================================================================

// In namespace ag::detail

void vjp_MSELoss(Node* n, const Tensor& gy){
    Node* Z_node = n->next_edges[0].function.get();
    Node* Y_node = n->next_edges[1].function.get();
    const float gy_scalar = gy.to_cpu().data<float>()[0];
    const float scale = 2.0f / static_cast<float>(Z_node->tensor.numel());
    Tensor diff = Z_node->tensor - Y_node->tensor;
    if (Z_node->requires_grad()) {
        std::lock_guard<std::mutex> lock(Z_node->mutex_);
        Z_node->tensor.grad_view() += (diff * (gy_scalar * scale));
    }
    if (Y_node->requires_grad()) {
        std::lock_guard<std::mutex> lock(Y_node->mutex_);
        Y_node->tensor.grad_view() += (diff * (-1.0f * gy_scalar * scale));
    }
}

// ===================================================================
// vjp_MAELoss
// ===================================================================
void vjp_MAELoss(Node* n, const Tensor& gy){
    Node* Z_node = n->next_edges[0].function.get();
    Node* Y_node = n->next_edges[1].function.get();
    const Tensor& Z = Z_node->tensor;
    const Tensor& Y = Y_node->tensor;

    const float inv_N = 1.0f / static_cast<float>(Z.numel());

    // Since OwnTensor doesn't have a sign op, we build it arithmetically
    const float epsilon = 1e-9f;
    Tensor diff = Z - Y;
    Tensor sign_diff = diff / (OwnTensor::abs(diff, ag::current_stream()) + epsilon);
    
    if (Z_node->requires_grad()) {
        std::lock_guard<std::mutex> lock(Z_node->mutex_);
        Z_node->tensor.grad_view() += gy * sign_diff * inv_N;
    }
    if (Y_node->requires_grad()) {
        std::lock_guard<std::mutex> lock(Y_node->mutex_);
        Y_node->tensor.grad_view() += gy * sign_diff * (-1.0f * inv_N);
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
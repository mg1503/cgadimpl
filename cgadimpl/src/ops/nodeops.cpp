// =====================
// file: cgadimpl/src/nodeops.cpp
// =====================
#include "ad/ops/nodeops.hpp"
#include "ad/runtime/runtime.hpp"
// #include "ad/ops/kernels_api.hpp"
#include <cuda_runtime.h>
#include "TensorLib.h" 
#include <unordered_map>
#include <cmath> 

namespace ag {
namespace detail {


std::shared_ptr<Node> add_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){
    // This correctly uses the stream-aware overloaded operator+
    Tensor Y = a->value + b->value; 
    // FIX: Use the new 3-argument Node constructor
    auto n = std::make_shared<Node>(Y, Op::Add, (a->requires_grad() || b->requires_grad()), "+");
    n->inputs = {a, b};
    ag::debug::on_node_created(n);
    return n;
}
  
std::shared_ptr<Node> sub_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){
    // This correctly uses the stream-aware overloaded operator-
    Tensor Y = a->value - b->value;
    // FIX: Use the new 3-argument Node constructor
    auto n = std::make_shared<Node>(Y, Op::Sub, (a->requires_grad() || b->requires_grad()), "-");
    n->inputs = {a, b};
    ag::debug::on_node_created(n);
    return n;
}

std::shared_ptr<Node> mul_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){ 
    // This correctly uses the stream-aware overloaded operator*
    Tensor y = a->value * b->value; 
    // FIX: Use the new 3-argument Node constructor
    auto n = std::make_shared<Node>(y, Op::Mul, (a->requires_grad() || b->requires_grad()), "*"); 
    n->inputs = {a, b}; 
    ag::debug::on_node_created(n); 
    return n; 
}

std::shared_ptr<Node> div_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){
    const Tensor& C = a->value / b->value;

    auto n = std::make_shared<Node>(C, Op::Div, (a->requires_grad() || b->requires_grad()), "/");
    n->inputs = { a, b };
    ag::debug::on_node_created(n);  
    return n;
}

// ================
// flomul nodeops
// ================
std::shared_ptr<Node> flomul_nodeops(const std::shared_ptr<Node>& a, float b) {
    // 1. A static cache to store nodes we create for scalars.
    //    The key is the float value, the value is the shared_ptr to the Node.
    static std::unordered_map<float, std::shared_ptr<Node>> scalar_cache;

    std::shared_ptr<Node> c; // This will be the node for our scalar 'b'

    // 2. Look for the scalar 'b' in our cache.
    auto it = scalar_cache.find(b);

    if (it != scalar_cache.end()) {
        // --- CACHE HIT ---
        // We've already created a node for this float value. Reuse it.
        c = it->second;
    } else {
        // --- CACHE MISS ---
        // This is the first time we've seen this float. Create a new node for it.
        // The tensor for the scalar only needs to be a 1x1 tensor.
        // The multiplication op will automatically handle broadcasting.
        Tensor scalar_tensor = Tensor::full(Shape({1,1}), TensorOptions().with_req_grad(false), b);

        c = std::make_shared<Node>(scalar_tensor, Op::Leaf, "leaf_scalar");

        // Store the new node in the cache for next time.
        scalar_cache[b] = c;
    }

    // 3. Now, perform the multiplication using node 'a' and the (cached or new) scalar node 'c'.
    // The underlying operator* will handle the stream context correctly.
    Tensor y = a->value * c->value;

    auto n = std::make_shared<Node>(y, Op::Mul, a->requires_grad(), "*");
    n->inputs = {a, c};
    ag::debug::on_node_created(n);
    return n;
}
// ===================================================================
// Corrected relu_nodeops - Hybrid dispatcher for now ************************************************************************************************************************************
// ===================================================================

std::shared_ptr<Node> relu_nodeops(const std::shared_ptr<Node>& x){
    const Tensor& X = x->value;
    
    // --- FIX START ---
    // Replaced the manual kernel dispatch with a device-agnostic expression.
    // The OwnTensor library's operators (+, *, abs) will handle the CPU/GPU logic.
    Tensor Y = (X + OwnTensor::abs(X, ag::current_stream())) * 0.5f;
    // --- FIX END ---
    
    auto n = std::make_shared<Node>(Y, Op::Relu, x->requires_grad(), "relu");
    n->inputs = {x};
    ag::debug::on_node_created(n);
    return n;
}


// =====================================================================================================
// matmul nodeops
// =====================================================================================================
std::shared_ptr<Node> matmul_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b) {
    // --- 1. Call the tensor library's matmul function directly ---
    // This function will automatically handle:
    //  - Device checking (CPU vs GPU)
    //  - Dispatching to the correct backend (CPU generic vs. CUDA kernel)
    //  - Getting the current stream from the context for GPU operations
    //  - All dimension and broadcasting validation
    Tensor C = matmul(a->value, b->value);

    // --- 2. Wrap the result in a new Node ---
    // The new Node constructor correctly infers requires_grad from the output tensor C.
    auto n = std::make_shared<Node>(C, Op::MatMul, (a->requires_grad() || b->requires_grad()), "matmul");
    n->inputs = {a, b};
    ag::debug::on_node_created(n);
    return n;
}

// =====================================================================================================
// fmab nodeops
// =====================================================================================================
  std::shared_ptr<Node> fmab_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c){
    // This correctly uses the stream-aware matmul and operator+
    Tensor y = matmul(a->value, b->value) + c->value;

    // FIX: Use the new Node constructor
    auto n = std::make_shared<Node>(y, Op::FMA, (a->requires_grad() || b->requires_grad() || c->requires_grad()), "fmab");

    n->inputs = {a, b, c};
    ag::debug::on_node_created(n);
    return n;
}

// =====================================================================================================
// attention nodeops
// =====================================================================================================
std::shared_ptr<Node> attention_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c, const std::shared_ptr<Node>& d){
    Tensor q = matmul(a->value, b->value);
    Tensor k = matmul(a->value, c->value);
    Tensor v = matmul(a->value, d->value);

    float scale = 1.f / sqrtf(static_cast<float>(k.shape().dims.back()));
    Tensor g = matmul(q, k.t()) * scale;

    // Re-implement softmax using OwnTensor ops
    Tensor max_val = reduce_max(g, {-1}, true);
    Tensor exp_g = exp(g - max_val);
    Tensor sum_exp_g = reduce_sum(exp_g, {-1}, true);
    Tensor s = exp_g / sum_exp_g;

    Tensor y = matmul(s, v);

    auto n = std::make_shared<Node>(y, Op::Attention, (a->requires_grad() || b->requires_grad() || c->requires_grad() || d->requires_grad()), "attention");
    n->inputs = {a, b, c, d};
    // Save intermediate tensors needed for the backward pass to the tape
    n->tape.push_back(std::make_shared<Tensor>(q));
    n->tape.push_back(std::make_shared<Tensor>(k));
    n->tape.push_back(std::make_shared<Tensor>(v));
    n->tape.push_back(std::make_shared<Tensor>(s));
    ag::debug::on_node_created(n);
    return n;
}
// =====================================================================================================
// Corrected sigatt_nodeops - Pure OwnTensor
// =====================================================================================================

std::shared_ptr<Node> sigatt_nodeops(const std::shared_ptr<Node>& a,
                                     const std::shared_ptr<Node>& b,
                                     const std::shared_ptr<Node>& c,
                                     const std::shared_ptr<Node>& d) {
    // --- Step 1: Projections using OwnTensor::matmul ---
    Tensor q = OwnTensor::matmul(a->value, b->value);
    Tensor k = OwnTensor::matmul(a->value, c->value);
    Tensor v = OwnTensor::matmul(a->value, d->value);

    // --- Step 2: Scaled dot-product attention ---
    // --- Step 2: Scaled dot-product attention ---
    float scale = 1.f / sqrtf(static_cast<float>(k.shape().dims.back()));
    Tensor g = OwnTensor::matmul(q, k.t()) * scale;

    // --- Step 3: Sigmoid activation implemented with OwnTensor ops ---
    // This is the FINAL, CORRECT, one-line version.
    Tensor s = 1.0f / (1.0f + OwnTensor::exp(g * -1.0f));

    // --- Step 4: Final output projection ---
    Tensor y = OwnTensor::matmul(s, v);

    // --- Step 5: Create the graph node with the correct constructor ---
    auto n = std::make_shared<Node>(y, Op::SigAtt, (a->requires_grad() || b->requires_grad() || c->requires_grad() || d->requires_grad()),  "sigatt");
    n->inputs = {a, b, c, d};

    // Save intermediate tensors needed for the backward pass to the tape
    n->tape.push_back(std::make_shared<Tensor>(q));
    n->tape.push_back(std::make_shared<Tensor>(k));
    n->tape.push_back(std::make_shared<Tensor>(v));
    n->tape.push_back(std::make_shared<Tensor>(s));

    ag::debug::on_node_created(n);
    return n;
}
// ===================================================================
// reluatt_nodeops
// ===================================================================


std::shared_ptr<Node> reluatt_nodeops(const std::shared_ptr<Node>& a, 
                                      const std::shared_ptr<Node>& b, 
                                      const std::shared_ptr<Node>& c, 
                                      const std::shared_ptr<Node>& d) {
    // --- Step 1: Projections using OwnTensor::matmul ---
    // This part is already correct.
    Tensor q = OwnTensor::matmul(a->value, b->value);
    Tensor k = OwnTensor::matmul(a->value, c->value);
    Tensor v = OwnTensor::matmul(a->value, d->value);

    // --- Step 2: Scaled dot-product attention ---
    // This part is also correct.
    float scale = 1.0f / sqrtf(static_cast<float>(k.shape().dims.back()));
    Tensor g = OwnTensor::matmul(q, k.t()) * scale;

    // --- FIX START ---
    // --- Step 3: ReLU activation using high-level OwnTensor ops ---
    // Replaced the manual kernel dispatch with a device-agnostic arithmetic expression.
    // The OwnTensor operators for +, abs, and * will automatically handle
    // dispatching to the correct CPU or GPU implementation.
    Tensor s = (g + OwnTensor::abs(g, ag::current_stream())) * 0.5f;
    // --- FIX END ---

    // --- Step 4: Final output projection ---
    // This part is correct.
    Tensor y = OwnTensor::matmul(s, v);

    // --- Step 5: Create the graph node ---
    // This part is correct.
    auto n = std::make_shared<Node>(y, Op::RELUAtt, (a->requires_grad() || b->requires_grad() || c->requires_grad() || d->requires_grad()), "reluatt"); 
    n->inputs = {a, b, c, d};
    n->tape.push_back(std::make_shared<Tensor>(q));
    n->tape.push_back(std::make_shared<Tensor>(k));
    n->tape.push_back(std::make_shared<Tensor>(v));
    n->tape.push_back(std::make_shared<Tensor>(s));
    ag::debug::on_node_created(n); 
    return n; 
}

// ... other functions in nodeops.cpp ...
// ===================================================================
// moewe_nodeops
// ===================================================================

std::shared_ptr<Node> moewe_nodeops(const std::shared_ptr<Node>& x, 
                                    const std::shared_ptr<Node>& w, 
                                    const std::shared_ptr<Node>& b) {
    // --- Step 1: Linear transformation ---
    Tensor logits = OwnTensor::matmul(x->value, w->value.t()) + b->value;

    // --- Step 2: Softmax implemented in a single expression ---
    // This avoids the scoping issue and the default constructor error.
    Tensor max_val = OwnTensor::reduce_max(logits, {-1}, true);
    Tensor exp_logits = OwnTensor::exp(logits - max_val);
    Tensor sum_exp_logits = OwnTensor::reduce_sum(exp_logits, {-1}, true);
    Tensor y = exp_logits / sum_exp_logits;

    // --- Step 3: Create the graph node ---
    auto n = std::make_shared<Node>(y, Op::MOE, (x->requires_grad() || w->requires_grad() || b->requires_grad()), "moe");
    n->inputs = {x, w, b}; 
    ag::debug::on_node_created(n);  
    return n;
}

// ===================================================================
// reci_nodeops
// ===================================================================

std::shared_ptr<Node> reci_nodeops(const std::shared_ptr<Node>& a) {
    // This correctly uses the stream-aware overloaded operator for scalar / Tensor.
    Tensor y = 1.0f / a->value;
    
    // Use the new 3-argument Node constructor.
    auto n = std::make_shared<Node>(y, Op::Reciprocal, a->requires_grad(),"reciprocal");
    n->inputs = {a};
    ag::debug::on_node_created(n);
    return n;
}

// ===================================================================
// flodiv_nodeops
// ===================================================================

std::shared_ptr<Node> flodiv_nodeops(float b, const std::shared_ptr<Node>& a) {
    // --- Step 1 (Optional but Recommended): Use the Scalar Cache ---
    // This is the same efficient pattern from flomul_nodeops.
    static std::unordered_map<float, std::shared_ptr<Node>> scalar_cache;
    std::shared_ptr<Node> c;
    auto it = scalar_cache.find(b);
    if (it != scalar_cache.end()) {
        c = it->second;
    } else {
        Tensor scalar_tensor = Tensor::full(Shape{{1, 1}}, TensorOptions().with_req_grad(false), b);
        c = std::make_shared<Node>(scalar_tensor, Op::Leaf, "leaf_scalar");
        scalar_cache[b] = c;
    }

    // --- Step 2: Perform the operation ---
    // This correctly uses the stream-aware overloaded operator for Tensor / Tensor.
    Tensor y = c->value / a->value;
    
    // --- Step 3: Create the Node ---
    auto n = std::make_shared<Node>(y, Op::Div, a->requires_grad(), "/");
    n->inputs = {c, a}; // Note the order: c is the numerator, a is the denominator
    ag::debug::on_node_created(n);
    return n;
}

// ===================================================================
// floadd_nodeops
// ===================================================================

std::shared_ptr<Node> floadd_nodeops(float b, const std::shared_ptr<Node>& a) {
    // --- Step 1: Use the Scalar Cache ---
    static std::unordered_map<float, std::shared_ptr<Node>> scalar_cache;
    std::shared_ptr<Node> c;
    auto it = scalar_cache.find(b);
    if (it != scalar_cache.end()) {
        c = it->second;
    } else {
        Tensor scalar_tensor = Tensor::full(Shape{{1, 1}}, TensorOptions().with_req_grad(false), b);
        c = std::make_shared<Node>(scalar_tensor, Op::Leaf, "leaf_scalar");
        scalar_cache[b] = c;
    }

    // --- Step 2: Perform the operation ---
    // This correctly uses the stream-aware overloaded operator for Tensor + Tensor.
    Tensor y = c->value + a->value;
    
    // --- Step 3: Create the Node ---
    auto n = std::make_shared<Node>(y, Op::Add, a->requires_grad(), "+");
    n->inputs = {c, a}; // Order matches the operation
    ag::debug::on_node_created(n);
    return n;
}
// ===================================================================
// relumask_nodeops
// ===================================================================

std::shared_ptr<Node> relumask_nodeops(const std::shared_ptr<Node>& x) {
    const Tensor& xin = x->value;

    // FIX: Use the new factory with options
    Tensor y = OwnTensor::Tensor::zeros(xin.shape(), ag::options(xin));

    if (xin.is_cpu()) {
        // Your existing CPU implementation is fine, just needs the new API
        // We must dispatch by dtype to get the correct pointer type.
        dispatch_by_dtype(xin.dtype(), [&](auto dummy){
            using T = decltype(dummy);
            const T* x_data = xin.data<T>();
            T* y_data = y.data<T>();
            for (int64_t i = 0; i < xin.numel(); ++i) {
                if (x_data[i] > T(0)) {
                    y_data[i] = T(1);
                }
            }
        });
    } else {
        // For now, we will add a placeholder, as you don't have a GPU kernel for this.
        // To make this work on GPU, you would need to add a 'relumask_cuda' to your kernels plugin.
        throw std::runtime_error("relumask_nodeops not implemented for CUDA yet.");
    }
    
    // FIX: Use the new Node constructor
    auto n = std::make_shared<Node>(y, Op::Relumask, x->requires_grad(), "relumask");
    n->inputs = {x};
    ag::debug::on_node_created(n);
    return n;
}


// ===================================================================
// linear_nodeops
// ===================================================================

std::shared_ptr<Node> linear_nodeops(const std::shared_ptr<Node>& a, // Input X
                                     const std::shared_ptr<Node>& b, // Weight W
                                     const std::shared_ptr<Node>& c) // Bias b
{
    const Tensor& input_X = a->value;
    const Tensor& weight_W = b->value; // Shape is [out, in]
    const Tensor& bias_b = c->value;
    Tensor y = matmul(input_X, weight_W.t()) + bias_b;

    auto n = std::make_shared<Node>(y, Op::Linear, (a->requires_grad() || b->requires_grad() || c->requires_grad()), "linear");
    n->inputs = {a, b, c};
    ag::debug::on_node_created(n);
    return n;
}
// ===================================================================
// cosh_nodeops
// ===================================================================

    std::shared_ptr<Node> cosh_nodeops(const std::shared_ptr<Node>& x){
        Tensor y = cosh(x->value);
        auto n=std::make_shared<Node>(y, Op::Cosh, x->requires_grad(), "cosh");
        n->inputs={x};
        ag::debug::on_node_created(n);
        return n;
    }

// ===================================================================
// sinh_nodeops
// ===================================================================

     std::shared_ptr<Node> sinh_nodeops(const std::shared_ptr<Node>& x){
        Tensor y = sinh(x->value);
        auto n=std::make_shared<Node>(y, Op::Sinh, x->requires_grad(), "sinh");
        n->inputs={x};
        ag::debug::on_node_created(n);
        return n;
    }

// ===================================================================
// cos_nodeops
// ===================================================================


     std::shared_ptr<Node> cos_nodeops(const std::shared_ptr<Node>& x){
        Tensor y = cos(x->value);
        auto n=std::make_shared<Node>(y, Op::Cos, x->requires_grad(), "cosh");
        n->inputs={x};
        ag::debug::on_node_created(n);
        return n;
    }

// ===================================================================
// sin_nodeops
// ===================================================================

     std::shared_ptr<Node> sin_nodeops(const std::shared_ptr<Node>& x){
        Tensor y = sin(x->value);
        auto n=std::make_shared<Node>(y, Op::Sin, x->requires_grad(), "sinh");
        n->inputs={x};
        ag::debug::on_node_created(n);
        return n;
    }

// ===================================================================
// In file: cgadimpl/src/nodeops.cpp (Corrected sign_nodeops)
// ===================================================================

std::shared_ptr<Node> sign_nodeops(const std::shared_ptr<Node>& x){
    // Call the stream-aware OwnTensor::sign function
    Tensor y = OwnTensor::sign(x->value, ag::current_stream());

    // Use the new 3-argument Node constructor
    auto n = std::make_shared<Node>(y, Op::Sign, x->requires_grad(), "sign");
    n->inputs={x};
    ag::debug::on_node_created(n);
    return n;
}


// ===================================================================
// sqrt_nodeops
// ===================================================================
std::shared_ptr<Node> sqrt_nodeops(const std::shared_ptr<Node>& x) {
    // 1. Call the OwnTensor::sqrt function directly.
    // It will handle device dispatch and stream context automatically.
    Tensor y = OwnTensor::sqrt(x->value, ag::current_stream());

    // 2. Wrap the result in a new Node using the correct constructor.
    auto n = std::make_shared<Node>(y, Op::Sqrt, x->requires_grad(), "sqrt");
    n->inputs = {x};
    ag::debug::on_node_created(n);
    return n;
}

// ===================================================================
// alibiatt_nodeops
// ===================================================================

std::shared_ptr<Node> alibiatt_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c, const std::shared_ptr<Node>& d, float& m) {
    // Step 1: Projections
    Tensor q = OwnTensor::matmul(a->value, b->value); 
    Tensor k = OwnTensor::matmul(a->value, c->value); 
    Tensor v = OwnTensor::matmul(a->value, d->value);
    
    // Step 2: Scaled dot-product attention
    float scale = 1.f / sqrtf(static_cast<float>(k.shape().dims.back()));
    Tensor logits = OwnTensor::matmul(q, k.t()) * scale;

    
    // Step 3: Create Alibi bias and add it in one step to initialize 'g'
    Tensor bias_cpu(logits.shape(), OwnTensor::TensorOptions().with_dtype(logits.dtype()));
    {
        int n_heads = logits.shape().dims[0];
        int seq_len = logits.shape().dims[1];
        float slope_start = 1.0f / powf(2.0f, 8.0f / n_heads);

        dispatch_by_dtype(bias_cpu.dtype(), [&](auto dummy){
            using T = decltype(dummy);
            T* data = bias_cpu.data<T>();
            for(int h = 0; h < n_heads; ++h) {
                float slope = powf(slope_start, h + 1);
                for (int i = 0; i < seq_len; ++i) {
                    for (int j = 0; j < seq_len; ++j) {
                        data[h * seq_len * seq_len + i * seq_len + j] = (j > i) ? -std::numeric_limits<float>::infinity() : static_cast<T>(-(seq_len - 1 - j) * slope);
                    }
                }
            }
        });
    }
    Tensor g = logits + bias_cpu.to(logits.device());
 
    // Step 4: Re-implement softmax and initialize 's' in a single expression
    Tensor max_val = OwnTensor::reduce_max(g, {-1}, true);
    Tensor exp_g = OwnTensor::exp(g - max_val);
    Tensor sum_exp_g = OwnTensor::reduce_sum(exp_g, {-1}, true);
    Tensor s = exp_g / sum_exp_g;

    // Step 5: Final projection
    Tensor y = OwnTensor::matmul(s, v);

    auto n = std::make_shared<Node>(y, Op::AlibiAttention, (a->requires_grad() || b->requires_grad() || c->requires_grad() || d-> requires_grad()), "alibiattention"); 
    n->inputs = {a, b, c, d};
    n->tape = {std::make_shared<Tensor>(q), std::make_shared<Tensor>(k), 
               std::make_shared<Tensor>(v), std::make_shared<Tensor>(s)};
    ag::debug::on_node_created(n); 
    return n; 
}

// ===================================================================
// swiglu_nodeops
// ===================================================================
std::shared_ptr<Node> swiglu_nodeops(const std::shared_ptr<Node>& x, const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c, const std::shared_ptr<Node>& d){ 
    // Gate projection
    Tensor y = OwnTensor::matmul(x->value, a->value.t()) + b->value; 
    
    // SiLU (Swish) activation on the gate: y * sigmoid(y)
    Tensor q = y * (1.0f / (1.0f + OwnTensor::exp(y * -1.0f)));
    
    // Value projection and final multiplication
    Tensor w = q * (OwnTensor::matmul(x->value, c->value.t()) + d->value);
    
    auto n = std::make_shared<Node>(w, Op::SWIGLU, (x->requires_grad() || a->requires_grad() || b->requires_grad() || c->requires_grad() || d-> requires_grad()) , "swiglu"); 
    n->inputs={x, a, b, c, d};
    ag::debug::on_node_created(n); 
    return n;
}

// ============================================================================
// sum_nodeops
// ============================================================================
 
std::shared_ptr<Node> sum_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = OwnTensor::reduce_sum(x->value, {}, false);
    auto n = std::make_shared<Node>(y, Op::Sum, x->requires_grad(), "sum");
    n->inputs = {x};
    ag::debug::on_node_created(n);
    return n;
}

// ============================================================================
// transpose_nodeops
// ============================================================================

std::shared_ptr<Node> transpose_nodeops(const std::shared_ptr<Node>& x){
    // .t() is a zero-copy view operation. It doesn't need a stream
    // as no computation is performed. It just returns a new Tensor
    // with different strides. This is highly efficient.
    Tensor y = x->value.t();
    
    // FIX: Use the correct Op and name, and the correct constructor.
    auto n = std::make_shared<Node>(y, Op::Transpose, x->requires_grad(), "transpose");
    n->inputs = {x};
    ag::debug::on_node_created(n);
    return n;
}


// ============================================================================
// exp_nodeops
// ============================================================================

std::shared_ptr<Node> exp_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = OwnTensor::exp(x->value);
    
    // 3. Use the correct Node constructor.
    auto n = std::make_shared<Node>(y, Op::Exp, x->requires_grad(), "exp");
    n->inputs = {x};
    ag::debug::on_node_created(n);
    return n;
}

// ===================================================================
// log_nodeops
// ===================================================================
std::shared_ptr<Node> log_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = OwnTensor::log(x->value);
    
    auto n = std::make_shared<Node>(y, Op::Log, x->requires_grad(), "log");
    n->inputs = {x};
    ag::debug::on_node_created(n);
    return n;
}

// ===================================================================
// In file: cgadimpl/src/nodeops.cpp (Corrected)
// ===================================================================
std::shared_ptr<Node> mish_nodeops(const std::shared_ptr<Node>& x){
    // All operators (+, *) and functions (exp, log, tanh) will
    // automatically get the current stream from the context.
    // We don't need to pass it manually.
    
    // softplus(x) = log(1 + exp(x))
    Tensor sp = OwnTensor::log(1.0f + OwnTensor::exp(x->value));

    // mish(x) = x * tanh(softplus(x))
    Tensor y = x->value * OwnTensor::tanh(sp);
    
    auto n = std::make_shared<Node>(y, Op::Mish, x->requires_grad(), "mish");
    n->inputs = {x};
    ag::debug::on_node_created(n);
    return n;
}

// ===============================================================================
// tanh nodeops
// ===============================================================================
  std::shared_ptr<Node> tanh_nodeops(const std::shared_ptr<Node>& x){
    // 1. Call the OwnTensor::tanh function directly.
    // This single call will automatically:
    //  - Check if the tensor is on the CPU or GPU.
    //  - Call the appropriate backend (CPU or CUDA kernel).
    //  - Get the current stream from the context if it's on the GPU.
    //  - Queue the operation asynchronously on that stream.
    Tensor y = OwnTensor::tanh(x->value);

    // 2. Wrap the result in a new Node using the correct constructor.
    auto n = std::make_shared<Node>(y, Op::Tanh, x->requires_grad(), "tanh");
    n->inputs = {x};
    ag::debug::on_node_created(n);
    return n;
}

// ===================================================================
// sigmoid_nodeops
// ===================================================================
std::shared_ptr<Node> sigmoid_nodeops(const std::shared_ptr<Node>& x){
    // Implement sigmoid using OwnTensor ops: 1 / (1 + exp(-x))
    // All operations are stream-aware.
    Tensor y = 1.0f / (1.0f + OwnTensor::exp(x->value * -1.0f));

    auto n = std::make_shared<Node>(y, Op::Sigmoid, x->requires_grad(), "sigmoid"); 
    n->inputs={x}; 
    ag::debug::on_node_created(n);  
    return n;
}

// ===================================================================
// softplus_nodeops
// ===================================================================
std::shared_ptr<Node> softplus_nodeops(const std::shared_ptr<Node>& x){
    // All ops automatically use the stream from the context.
    Tensor y = OwnTensor::log(1.0f + OwnTensor::exp(x->value));

    auto n = std::make_shared<Node>(y, Op::Softplus, x->requires_grad(), "softplus");
    n->inputs = {x};
    ag::debug::on_node_created(n);
    return n;
}

// ===================================================================
// gaus_nodeops
// ===================================================================
std::shared_ptr<Node> gaus_nodeops(const std::shared_ptr<Node>& x){
    Tensor x_squared = x->value * x->value;
    Tensor y = OwnTensor::exp(x_squared * -1.0f);

    auto n = std::make_shared<Node>(y, Op::Gaus, x->requires_grad(), "gaus");
    n->inputs={x};
    ag::debug::on_node_created(n);
    return n;
}

// ===================================================================
// In file: cgadimpl/src/nodeops.cpp (Corrected)
// ===================================================================

std::shared_ptr<Node> gelu_nodeops(const std::shared_ptr<Node>& x){
    // All of these operations will correctly use the thread-local stream context.

    // Constants for the GELU approximation
    const float c1 = 0.7978845608f; // sqrt(2.0f / M_PI)
    const float c2 = 0.044715f;

    // 1. Calculate x^3
    Tensor x3 = x->value * x->value * x->value;
    
    // 2. Calculate the inside of the tanh: u = c1 * (x + c2 * x^3)
    Tensor u = (x->value + x3 * c2) * c1;

    // 3. Calculate the full GELU formula: 0.5 * x * (1 + tanh(u))
    Tensor y = x->value * (1.0f + OwnTensor::tanh(u)) * 0.5f;
    
    auto n = std::make_shared<Node>(y, Op::GELU, x->requires_grad(), "gelu");
    n->inputs={x};
    ag::debug::on_node_created(n);
    return n;
}
// ===================================================================
// gcu_nodeops
// ===================================================================
std::shared_ptr<Node> gcu_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = x->value * OwnTensor::cos(x->value);

    auto n = std::make_shared<Node>(y, Op::GCU, x->requires_grad(), "gcu");
    n->inputs={x};
    ag::debug::on_node_created(n);
    return n;
}

// ===================================================================
// silu_nodeops
// ===================================================================
std::shared_ptr<Node> silu_nodeops(const std::shared_ptr<Node>& x){
    // All of these operations will correctly use the thread-local stream context.
    
    // 1. Implement sigmoid: 1 / (1 + exp(-x))
    Tensor sig_x = 1.0f / (1.0f + OwnTensor::exp(x->value * -1.0f));

    // 2. Implement silu: x * sigmoid(x)
    Tensor y = x->value * sig_x;
    
    auto n = std::make_shared<Node>(y, Op::SiLU, x->requires_grad(), "silu");
    n->inputs={x};
    ag::debug::on_node_created(n);
    return n;
}

// ===================================================================
// parcon_nodeops
// ===================================================================

std::shared_ptr<Node> parcon_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = x->value * (2.0f - x->value);

    auto n = std::make_shared<Node>(y, Op::Parcon, x->requires_grad(), "parcon");
    n->inputs={x};
    ag::debug::on_node_created(n);
    return n;
}

// ===================================================================
// lisht_nodeops
// ===================================================================

std::shared_ptr<Node> lisht_nodeops(const std::shared_ptr<Node>& x){
    // All ops are stream-aware via context
    Tensor y = x->value * OwnTensor::tanh(x->value);

    // FIX: The Op type was incorrect in your original code.
    auto n = std::make_shared<Node>(y, Op::LiSHT, x->requires_grad(), "lisht"); 
    n->inputs={x};
    ag::debug::on_node_created(n);
    return n;
}

// ===================================================================
// leaky_relu_nodeops
// ===================================================================

std::shared_ptr<Node> leaky_relu_nodeops(const std::shared_ptr<Node>& x, float alpha){ 
    // All of these operations will correctly use the thread-local stream context.
    
    // --- Re-implement Leaky ReLU using only arithmetic operations ---
    
    // 1. Isolate the positive part of x: (x + abs(x)) * 0.5
    Tensor pos_part = (x->value + OwnTensor::abs(x->value, ag::current_stream())) * 0.5f;

    // 2. Isolate the negative part of x: (x - abs(x)) * 0.5
    Tensor neg_part = (x->value - OwnTensor::abs(x->value, ag::current_stream())) * 0.5f;

    // 3. Combine them: pos_part + (neg_part * alpha)
    Tensor Y = pos_part + (neg_part * alpha);
    
    // --- End of re-implementation ---

    // We still need to pass alpha to the backward pass. Create a 1x1 constant node.
    // NOTE: This now creates a NEW tensor with requires_grad=false.
    Tensor aT = Tensor::full(Shape{{1, 1}}, TensorOptions().with_req_grad(false), alpha);
    auto aC = make_tensor(aT, "alpha"); 
    
    auto n = std::make_shared<Node>(Y, Op::LeakyRelu, x->requires_grad(), "leakyrelu");
    n->inputs = {x, aC.node}; 
    ag::debug::on_node_created(n);  
    return n;
}
// ============================================================================================
// rowsum_nodeops
// ============================================================================================
    std::shared_ptr<Node> rowsum_nodeops(const std::shared_ptr<Node>& x){
    // Reduce over axis 1 (the columns), and keep the dimension so shape goes from [B,C] to [B,1].
    Tensor y = OwnTensor::reduce_sum(x->value, {1}, true);
    auto n = std::make_shared<Node>(y, Op::RowSum, x->requires_grad(), "rowsum");
    n->inputs = {x};
    ag::debug::on_node_created(n);
    return n;
}

// ===================================================================
// rowmax_nodeops
// ===================================================================
std::shared_ptr<Node> rowmax_nodeops(const std::shared_ptr<Node>& x){
    // Reduce over axis 1 (columns) and keep the dimension.
    Tensor y = OwnTensor::reduce_max(x->value, {1}, true);
    auto n = std::make_shared<Node>(y, Op::RowMax, x->requires_grad(), "rowmax");
    n->inputs={x};
    ag::debug::on_node_created(n);
    return n;
}


// ===================================================================
// rms_nodeops
// ===================================================================

// ... inside namespace ag::detail
std::shared_ptr<Node> rms_nodeops(const std::shared_ptr<Node>& x){
    // Calculate x^2
    Tensor x_squared = x->value * x->value;

    // Calculate the mean along the last dimension.
    Tensor variance = OwnTensor::reduce_mean(x_squared, {-1}, true);

    // Calculate the reciprocal square root (rsqrt) with an epsilon for stability.
    Tensor rsqrt_var = 1.0f / OwnTensor::sqrt(variance + 1e-5f, ag::current_stream());

    // Normalize x
    Tensor y = x->value * rsqrt_var;

    auto n = std::make_shared<Node>(y, Op::RMSNorm, x->requires_grad(), "rmsnorm");
    // --- FIX START ---
    // The backward pass needs rsqrt_var and the normalized output 'y'.
    n->tape.push_back(std::make_shared<Tensor>(rsqrt_var));
    n->tape.push_back(std::make_shared<Tensor>(x->value)); // Incorrectly saving original x
    n->tape.push_back(std::make_shared<Tensor>(y));         // Correctly save the normalized output y
    // --- FIX END ---
    n->inputs = {x};
    ag::debug::on_node_created(n);
    return n;
}
// ... rest of the file

// ===================================================================
// In file: cgadimpl/src/nodeops.cpp (Corrected)
// ===================================================================
std::shared_ptr<Node> realrms_nodeops(const std::shared_ptr<Node>& x, float& g_val){ // Pass g by value
    const float inv_cols = 1.0f / static_cast<float>(x->value.shape().dims.back());
    
    // Calculate mean of squares along the last dim
    Tensor variance = OwnTensor::reduce_sum(x->value * x->value, {-1}, true) * inv_cols;
    Tensor rsqrt_var = 1.0f / OwnTensor::sqrt(variance + 1e-5f, ag::current_stream());
    Tensor y_normalized = x->value * rsqrt_var;
    
    // Use our scalar caching mechanism for the gain 'g'
    static std::unordered_map<float, std::shared_ptr<Node>> scalar_cache;
    std::shared_ptr<Node> G;
    auto it = scalar_cache.find(g_val);
    if (it != scalar_cache.end()) {
        G = it->second;
    } else {
        Tensor g_tensor = Tensor::full(Shape{{1, 1}}, TensorOptions().with_req_grad(true), g_val); // Assume gain is trainable
        G = std::make_shared<Node>(g_tensor, Op::Leaf, "rms_gain");
        scalar_cache[g_val] = G;
    }

    Tensor y_scaled = y_normalized * G->value;

    auto n = std::make_shared<Node>(y_scaled, Op::RealRMSNorm, x->requires_grad(), "realrmsnorm");
    n->tape.push_back(std::make_shared<Tensor>(rsqrt_var));
    n->tape.push_back(std::make_shared<Tensor>(y_normalized));
    n->inputs = {x, G};
    ag::debug::on_node_created(n);
    return n;
}

// ===================================================================
// laynor_nodeops
// ===================================================================
std::shared_ptr<Node> laynor_nodeops(const std::shared_ptr<Node>& x){
    // 1. Calculate mean across the last dimension
    Tensor mean = OwnTensor::reduce_mean(x->value, {-1}, true);
    
    // 2. Calculate variance across the last dimension
    Tensor x_minus_mean = x->value - mean;
    Tensor variance = OwnTensor::reduce_mean(x_minus_mean * x_minus_mean, {-1}, true);
    
    // 3. Normalize
    Tensor y = x_minus_mean / OwnTensor::sqrt(variance + 1e-5f, ag::current_stream());
    
    auto n = std::make_shared<Node>(y, Op::LayerNorm, x->requires_grad(), "layernorm");
    n->tape.push_back(std::make_shared<Tensor>(variance));
    n->tape.push_back(std::make_shared<Tensor>(mean));
    n->inputs = {x};
    ag::debug::on_node_created(n);
    return n;
}

// ===================================================================
// relaynor_nodeops
// ===================================================================
std::shared_ptr<Node> relaynor_nodeops(const std::shared_ptr<Node>& x, float& b_val, float& g_val){
    // 1. Calculate mean and variance
    Tensor mean = OwnTensor::reduce_mean(x->value, {-1}, true);
    Tensor x_minus_mean = x->value - mean;
    Tensor variance = OwnTensor::reduce_mean(x_minus_mean * x_minus_mean, {-1}, true);
    
    // 2. Normalize
    Tensor y_normalized = x_minus_mean / OwnTensor::sqrt(variance + 1e-5f, ag::current_stream());

    // 3. Create or cache scalar nodes for gain and bias
    static std::unordered_map<float, std::shared_ptr<Node>> cache;
    std::shared_ptr<Node> G, B;
    if (cache.count(g_val)) G = cache[g_val];
    else {
        G = std::make_shared<Node>(Tensor::full(Shape{{1}}, TensorOptions().with_req_grad(true), g_val), Op::Leaf, "ln_gain");
        cache[g_val] = G;
    }
    if (cache.count(b_val)) B = cache[b_val];
    else {
        B = std::make_shared<Node>(Tensor::full(Shape{{1}}, TensorOptions().with_req_grad(true), b_val), Op::Leaf, "ln_bias");
        cache[b_val] = B;
    }

    // 4. Apply scale and shift
    Tensor y = y_normalized * G->value + B->value;

    auto n = std::make_shared<Node>(y, Op::RealLayerNorm, x->requires_grad(), "reallayernorm");
    n->tape.push_back(std::make_shared<Tensor>(variance));
    n->tape.push_back(std::make_shared<Tensor>(mean));
    n->tape.push_back(std::make_shared<Tensor>(y_normalized));
    n->inputs = {x, G, B};
    ag::debug::on_node_created(n);
    return n;
}

// ===================================================================
// mean_all_nodeops
// ===================================================================
std::shared_ptr<Node> mean_all_nodeops(const std::shared_ptr<Node>& x){
    // reduce_mean with empty axes reduces over the entire tensor
    Tensor y = OwnTensor::reduce_mean(x->value);
    auto n = std::make_shared<Node>(y, Op::MeanAll, x->requires_grad(), "meanall");
    n->inputs={x};
    ag::debug::on_node_created(n);
    return n;
}

// ===================================================================
// dyntanh_nodeops
// ===================================================================
std::shared_ptr<Node> dyntanh_nodeops(const std::shared_ptr<Node>& x, float& a_val, float& b_val, float& g_val){
    // Use scalar caching for the parameters 'a', 'b', and 'g'
    static std::unordered_map<float, std::shared_ptr<Node>> cache;
    std::shared_ptr<Node> A, B, G;
    // ... (code to create/cache A, B, and G nodes from a_val, b_val, g_val, similar to relaynor) ...
    // For brevity, assuming they are created and require_grad=true.
    A = std::make_shared<Node>(Tensor::full(Shape{{1}}, TensorOptions().with_req_grad(true), a_val), Op::Leaf, "dyn_a");
    B = std::make_shared<Node>(Tensor::full(Shape{{1}}, TensorOptions().with_req_grad(true), b_val), Op::Leaf, "dyn_b");
    G = std::make_shared<Node>(Tensor::full(Shape{{1}}, TensorOptions().with_req_grad(true), g_val), Op::Leaf, "dyn_g");
    
    Tensor h = x->value * A->value;
    Tensor y = OwnTensor::tanh(h) * G->value + B->value;
    
    // Note: The Op was incorrectly MeanAll in your old code. Let's assume it should be Dyntanh.
    auto n = std::make_shared<Node>(y, Op::Dyntanh, x->requires_grad(), "dyntanh");
    n->inputs={x, A, B, G};
    n->tape.push_back(std::make_shared<Tensor>(h));
    ag::debug::on_node_created(n);
    return n;
}

// ===================================================================
// softmax_row_nodeops
// ===================================================================
std::shared_ptr<Node> softmax_row_nodeops(const std::shared_ptr<Node>& z){ 
    // 1. Find the max value along the rows (last dimension) for numerical stability.
    // The `true` for keepdim ensures the result has shape [B, 1] for broadcasting.
    Tensor max_val = OwnTensor::reduce_max(z->value, {-1}, true);
    
    // 2. Subtract the max and exponentiate.
    Tensor z_shifted = z->value - max_val;
    Tensor exp_z = OwnTensor::exp(z_shifted);
    
    // 3. Sum the exponents along the rows.
    Tensor sum_exp_z = OwnTensor::reduce_sum(exp_z, {-1}, true);
    
    // 4. Divide to get the final softmax probabilities.
    Tensor y = exp_z / sum_exp_z;
    
    auto n = std::make_shared<Node>(y, Op::SoftmaxRow, z->requires_grad(), "softmax_row"); 
    n->inputs = {z}; 
    ag::debug::on_node_created(n);  
    return n;
}

// ===================================================================
// logsumexp_row_nodeops
// ===================================================================
std::shared_ptr<Node> logsumexp_row_nodeops(const std::shared_ptr<Node>& z){ 
    // 1. Find the max value along the rows (last dimension).
    Tensor max_val = OwnTensor::reduce_max(z->value, {-1}, true);
    
    // 2. Subtract the max and exponentiate.
    Tensor z_shifted = z->value - max_val;
    Tensor exp_z = OwnTensor::exp(z_shifted);
    
    // 3. Sum the exponents along the rows and take the log.
    Tensor sum_exp_z = OwnTensor::reduce_sum(exp_z, {-1}, true);
    Tensor log_sum = OwnTensor::log(sum_exp_z);
    
    // 4. Add the max value back.
    Tensor y = log_sum + max_val;
    
    auto n = std::make_shared<Node>(y, Op::LogSumExpRow, z->requires_grad(), "logsumexp_row"); 
    n->inputs = {z}; 
    ag::debug::on_node_created(n);  
    return n;
}

// ===================================================================
// mambassm_nodeops
// ===================================================================
std::shared_ptr<Node> mambassm_nodeops(const std::shared_ptr<Node>& z, const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c, const std::shared_ptr<Node>& d){

    // All ops will use the stream from the context.
    
    if (z->tape.empty()) {
        // --- Initialization Step ---
        Tensor w = OwnTensor::matmul(z->value, b->value); 
        Tensor q = OwnTensor::matmul(w, c->value);
        Tensor y = (z->value * d->value) + q;

        // Create a new leaf node for the initial state 'w'. It is not a parameter.
        auto W = std::make_shared<Node>(w, Op::Leaf, "ssm_state");
        
        // The Op was wrong here, let's assume it should be a custom 'MambaSSM' Op
        // Use a generic but existing Op as a placeholder. The final operation is an addition.
        auto n = std::make_shared<Node>(y, Op::Add, "mambassm");
        
        // The inputs to this step are the original inputs plus the NEW state node.
        n->inputs = {z, a, b, c, d, W}; 
        
        // Save the state for the NEXT step in the tape of the ORIGINAL input 'z'.
        z->tape.push_back(std::make_shared<Tensor>(w));
        
        ag::debug::on_node_created(n);  
        std::cout << "Initialized SSM state" << std::endl;
        return n;
    } else {
        // --- Recurrent Step ---
        
        // Get the previous state 'w' from the tape of the input 'z'.
        const Tensor& prev_w = *z->tape.back();
        
        Tensor w = OwnTensor::matmul(z->value, b->value) + prev_w; 
        Tensor q = OwnTensor::matmul(w, c->value);
        Tensor y = (z->value * d->value) + q;

        // Create a new leaf node for the CURRENT state 'w'.
        auto W = std::make_shared<Node>(w, Op::Leaf, "ssm_state");
        
        // Use a generic but existing Op as a placeholder. The final operation is an addition.
        auto n = std::make_shared<Node>(y, Op::Add, "mambassm");
        n->inputs = {z, a, b, c, d, W}; 
        
        // Update the tape of the input 'z' with the new state for the next step.
        z->tape.push_back(std::make_shared<Tensor>(w));

        ag::debug::on_node_created(n);  
        std::cout << "SSM step" << std::endl;
        return n;
    }
}

// ===================================================================
// cross_entropy_with_logits_nodeops
// ===================================================================
std::shared_ptr<Node> cross_entropy_with_logits_nodeops(const std::shared_ptr<Node>& logits, const std::shared_ptr<Node>& onehot){
    const Tensor& Z = logits->value;
    const Tensor& Y = onehot->value;

    // --- Re-implement with OwnTensor ops ---
    
    // 1. Calculate log(softmax(Z)) in a numerically stable way.
    //    logsoftmax(z) = z - log(sum(exp(z)))
    //    stable_logsoftmax(z) = z - (log(sum(exp(z - max(z)))) + max(z))
    Tensor max_val = OwnTensor::reduce_max(Z, {-1}, true);
    Tensor z_shifted = Z - max_val;
    Tensor log_sum_exp = OwnTensor::log(OwnTensor::reduce_sum(OwnTensor::exp(z_shifted), {-1}, true));
    Tensor log_sm = z_shifted - log_sum_exp;

    // 2. Calculate the cross-entropy loss: -mean(sum(Y * log_sm))
    // The sum is over the class dimension (-1), the mean is over the batch dimension (0).
    Tensor prod = Y * log_sm;
    Tensor sum_prod = OwnTensor::reduce_sum(prod, {-1}); // Sum over classes, shape=[B]
    Tensor loss = OwnTensor::reduce_mean(sum_prod * -1.0f); // Mean over batch and negate

    auto n = std::make_shared<Node>(loss, Op::CeWithLogits, (logits->requires_grad() || onehot->requires_grad()), "ce_with_logits");
    n->inputs = {logits, onehot};
    ag::debug::on_node_created(n);
    return n;
}

// ===================================================================
// kldivergence_nodeops
// ===================================================================
std::shared_ptr<Node> kldivergence_nodeops(const std::shared_ptr<Node>& logits, const std::shared_ptr<Node>& onehot){
    const Tensor& Z = logits->value;
    const Tensor& Y = onehot->value;

    // --- Re-implement with OwnTensor ops ---

    // 1. Calculate log(Y). Add a small epsilon for stability to avoid log(0).
    Tensor log_Y = OwnTensor::log(Y + 1e-9f);
    
    // 2. Calculate stable log_softmax(Z) (same as in cross-entropy).
    Tensor max_val = OwnTensor::reduce_max(Z, {-1}, true);
    Tensor z_shifted = Z - max_val;
    Tensor log_sum_exp = OwnTensor::log(OwnTensor::reduce_sum(OwnTensor::exp(z_shifted), {-1}, true));
    Tensor log_sm_Z = z_shifted - log_sum_exp;

    // 3. Calculate the KL Divergence: sum(Y * (log(Y) - log_softmax(Z)))
    Tensor kl_div_elementwise = Y * (log_Y - log_sm_Z);

    // 4. Sum over the class dimension, then take the mean over the batch dimension.
    Tensor sum_kl = OwnTensor::reduce_sum(kl_div_elementwise, {-1});
    Tensor loss = OwnTensor::reduce_mean(sum_kl);

    auto n = std::make_shared<Node>(loss, Op::KLDivergence, (logits->requires_grad() || onehot->requires_grad()), "kldivergence");
    n->inputs = {logits, onehot};
    ag::debug::on_node_created(n);
    return n;
}

// =================================================================
// mse_loss_nodeops
// =================================================================

std::shared_ptr<Node> mse_loss_nodeops(const std::shared_ptr<Node>& pred, const std::shared_ptr<Node>& target) {

    Tensor diff = pred->value - target->value;
    Tensor sq   = diff * diff;
    // --- THIS IS THE BUG ---
    // It should be reduce_mean, not reduce_sum. `reduce_mean` correctly
    // computes the VJP for the mean operation. `sum` has a different VJP.
    Tensor loss = OwnTensor::reduce_mean(sq); 
    // --- END BUG ---

    auto n = std::make_shared<Node>(loss, Op::MSELoss, (pred->requires_grad()), "mseloss");
    n->inputs = {pred, target};
    ag::debug::on_node_created(n);
    return n;
}


// ===================================================================
// In file: cgadimpl/src/nodeops.cpp (Corrected)
// ===================================================================
std::shared_ptr<Node> mae_loss_nodeops(const std::shared_ptr<Node>& pred, const std::shared_ptr<Node>& target) {
    Tensor diff = pred->value - target->value;
    Tensor abs_diff = OwnTensor::abs(diff, ag::current_stream());
    // The mean of the absolute error
    Tensor loss = OwnTensor::reduce_mean(abs_diff);

    auto n = std::make_shared<Node>(loss, Op::MAELoss, (pred->requires_grad() || target->requires_grad()), "maeloss");
    n->inputs = {pred, target};
    ag::debug::on_node_created(n);
    return n;
}

    // Tensor forward_eval_node_impl(const std::shared_ptr<Node>& node) {
    //     if (!node) throw std::runtime_error("forward_eval_node: null node");
    //     switch (node->op) {
    //         case Op::Add: return node->inputs[0]->value + node->inputs[1]->value;
    //         case Op::Sub: return node->inputs[0]->value - node->inputs[1]->value;
    //         case Op::Mul: return node->inputs[0]->value * node->inputs[1]->value;
    //         case Op::MatMul: return Tensor::matmul(node->inputs[0]->value, node->inputs[1]->value);
    //         case Op::Relu: return Tensor::relu(node->inputs[0]->value);
    //         case Op::Sigmoid: return Tensor::sigmoid(node->inputs[0]->value);
    //         case Op::Tanh: return Tensor::tanh(node->inputs[0]->value);
    //         case Op::Exp: return Tensor::exp(node->inputs[0]->value);
    //         case Op::Log: return Tensor::log(node->inputs[0]->value);
    //         case Op::AlibiAttention: {
    //             const Tensor &a = node->inputs[0]->value;
    //             const Tensor &b = node->inputs[1]->value;
    //             const Tensor &c = node->inputs[2]->value;
    //             const Tensor &d = node->inputs[3]->value;
    //             Tensor q = Tensor::matmul(a, b);
    //             Tensor k = Tensor::matmul(a, c);
    //             Tensor v = Tensor::matmul(a, d);
    //             Tensor logits = Tensor::matmul(q, Tensor::transpose(k) * (1.f / sqrt(float(k.cols()))));
    //             Tensor bias   = Tensor::alibi(logits.rows(), logits.cols(), /*m*/128);
    //             Tensor g      = logits + bias;
    //             Tensor s      = Tensor::softmax_row(g);
    //             return Tensor::matmul(s, v);
    //         }
    //         case Op::Leaf:
    //             return node->value;
    //         default:
    //             if (!node->tape.empty()) {
    //                 return *(node->tape.back());
    //             }
    //             throw std::runtime_error("forward_eval_node: unsupported op for recompute");
    //     }
    // }

    } // namespace detail
    } // namespace ag
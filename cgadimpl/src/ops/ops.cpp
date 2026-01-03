// =====================
// file: cgadimpl/src/ops.cpp
// =====================
#include "ad/ops/ops.hpp"
#include "ad/ops/nodeops.hpp" // Include the new node-level declarations
#include "ad/autodiff/inplace.hpp"
#include "ad/ops/nodeops.hpp"
#include <cuda_runtime.h>
#include "tensor.hpp" 
#include <unordered_map>
#include <cmath> 
#include <type_traits> 

namespace ag {

    Value checkpoint(const Value &v, const CheckpointOptions &opts) {
    if (!v.node) return v;
    ag::checkpoint_impl::mark_node_checkpoint(v.node, opts);
    return v;
}
    Value inplace_checkpoint(const Value& v) {
        if (!v.node) return v;
        ag::inplace::mark_inplace_checkpoint(v.node);
        return v;
    }

// Binary arith -----------------
    Value add(const Value& a, const Value& b){ 
        return Value(ag::detail::add_nodeops(a.node, b.node)); 
    }
    Value sub(const Value& a, const Value& b){ 
        return Value(ag::detail::sub_nodeops(a.node, b.node)); 
    }
    Value mul(const Value& a, const Value& b){ 
        return Value(ag::detail::mul_nodeops(a.node, b.node)); 
    }
    Value div(const Value& a, const Value& b){ 
        return Value(ag::detail::div_nodeops(a.node, b.node)); 
    }

// Classic Activations ---------------
    Value relu(const Value& x){ 
        return Value(ag::detail::relu_nodeops(x.node));
    }
    Value sigmoid(const Value& x){ 
        return Value(ag::detail::sigmoid_nodeops(x.node));
    }
    Value tanh(const Value& x){ 
        return Value(ag::detail::tanh_nodeops(x.node));
    }
    Value softplus(const Value& x){ 
        return Value(ag::detail::softplus_nodeops(x.node));
    }

// Smooth Activations (better gradient flow) ---
    Value gelu(const Value& x){ 
        return Value(ag::detail::gelu_nodeops(x.node));
    }
    Value silu(const Value& x){ 
        return Value(ag::detail::silu_nodeops(x.node));
    }
    Value mish(const Value& x){ 
        return Value(ag::detail::mish_nodeops(x.node));
    }

// Parametric Activations ------
    Value leaky_relu(const Value& x, float alpha){ 
        return Value(ag::detail::leaky_relu_nodeops(x.node, alpha));
    }

// Specialized activations -----------
    Value gaus(const Value& x){ 
        return Value(ag::detail::gaus_nodeops(x.node));
    }
    Value parcon(const Value& x){ 
        return Value(ag::detail::parcon_nodeops(x.node));
    }
    Value lisht(const Value& x){ 
        return Value(ag::detail::lisht_nodeops(x.node));
    }
// Standard Attention ---------
    Value attention(const Value& a, const Value& b, const Value& c, const Value& d){ 
        return Value(ag::detail::attention_nodeops(a.node, b.node, c.node, d.node));
    }
// Gated activation -----------------
    Value swiglu(const Value& x, const Value& a, const Value& b, const Value& c, const Value& d){ 
        return Value(ag::detail::swiglu_nodeops(x.node, a.node, b.node, c.node, d.node));
    }
//Leaf -----------

//Unary Mathematical Functions ------------------
    Value exp(const Value& x){ 
        return Value(ag::detail::exp_nodeops(x.node));
    }
    Value log(const Value& x){ 
        return Value(ag::detail::log_nodeops(x.node));
    }
    Value sqrt(const Value& x){
        return Value(ag::detail::sqrt_nodeops(x.node));
    }
    Value reci(const Value& x){ 
        return Value(ag::detail::reci_nodeops(x.node));
    }
    Value sign(const Value& a){ 
        return Value(ag::detail::sign_nodeops(a.node)); 
    }
    Value abs(const Value& x){
        return Value(ag::detail::abs_nodeops(x.node));
    }
    Value pow(const Value& a, const Value& b){ 
        return Value(ag::detail::pow_nodeops(a.node, b.node)); 
    }

//Core Matrix Operations ------------------------
    Value matmul(const Value& a, const Value& b){ 
         return Value(ag::detail::matmul_nodeops(a.node, b.node)); 
    }
    Value transpose(const Value& x){ 
        return Value(ag::detail::transpose_nodeops(x.node));
    }

//Fused Operations (better performance, fewer memory accesses) ---------------
    Value linear(const Value& a, const Value& b, const Value& c){ 
        return Value(ag::detail::linear_nodeops(a.node, b.node, c.node)); 
    }
    Value fmab(const Value& a, const Value& b, const Value& c){ 
        return Value(ag::detail::fmab_nodeops(a.node, b.node, c.node)); 
    }

//Classification losses ---------------
    Value cross_entropy_with_logits(const Value& logits, const Value& onehot){
        return Value(ag::detail::cross_entropy_with_logits_nodeops(logits.node, onehot.node));
    }
    Value kldivergence(const Value& logits, const Value& onehot){
        return Value(ag::detail::kldivergence_nodeops(logits.node, onehot.node));
    }

//Regression Losses --------------
    Value mse_loss(const Value& pred, const Value& target) {
        return Value(ag::detail::mse_loss_nodeops(pred.node, target.node));
    }
    Value mae_loss(const Value& pred, const Value& target) {
        return Value(ag::detail::mae_loss_nodeops(pred.node, target.node));
    }

//Layer Normalization ------------
    Value laynor(const Value& x){ 
        return Value(ag::detail::laynor_nodeops(x.node));
    }

//RMS Normalization -------------
    Value rms(const Value& x){ 
        return Value(ag::detail::rms_nodeops(x.node));
    }
    Value realrms(const Value& x, float g){ 
        return Value(ag::detail::realrms_nodeops(x.node, g));
    }

//Dynamic Normalization --------------
    Value dyntanh(const Value& x, float a, float b, float g){ 
        return Value(ag::detail::dyntanh_nodeops(x.node, a, b, g));
    }


//Global Reductions -------------------
    Value sum(const Value& x){ 
        return Value(ag::detail::sum_nodeops(x.node));
    }
    Value mean_all(const Value& x){ 
        return Value(ag::detail::mean_all_nodeops(x.node));
    }

//Row-wise Reductions ------------------------
    Value rowsum(const Value& x){ 
        return Value(ag::detail::rowsum_nodeops(x.node));
    }
    Value rowmax(const Value& x){ 
        return Value(ag::detail::rowmax_nodeops(x.node));
    }

//Softmax Family ---------------
    Value softmax_row(const Value& z){ 
        return Value(ag::detail::softmax_row_nodeops(z.node));
    }
    Value logsumexp_row(const Value& z){ 
        return Value(ag::detail::logsumexp_row_nodeops(z.node));
    }

//Trigonometric Functions --------------------
    Value sin(const Value& x){
        return Value(ag::detail::sin_nodeops(x.node));
    }
    Value cos(const Value& x){ 
        return Value(ag::detail::cos_nodeops(x.node));
    }
    Value tan(const Value& x){ 
        return Value(ag::detail::tan_nodeops(x.node));
    }


//Hyperbolic Functions ------------------
    Value cosh(const Value& x){ 
        return Value(ag::detail::cosh_nodeops(x.node));
    }
    Value sinh(const Value& x){ 
        return Value(ag::detail::sinh_nodeops(x.node));
    }

//Inverse Trigonometric Functions --------------
    Value asin(const Value& x){ 
        return Value(ag::detail::asin_nodeops(x.node));
    }
    Value acos(const Value& x){ 
        return Value(ag::detail::acos_nodeops(x.node));
    }
    Value atan(const Value& x){ 
        return Value(ag::detail::atan_nodeops(x.node));
    }

//Inverse Hyperbolic Trigonometric Functions ----------------
    Value asinh(const Value& x){ 
        return Value(ag::detail::asinh_nodeops(x.node));
    }
    Value acosh(const Value& x){ 
        return Value(ag::detail::acosh_nodeops(x.node));
    }
    Value atanh(const Value& x){ 
        return Value(ag::detail::atanh_nodeops(x.node));
    }

    







    


    





//  The implementation of **forward evaluation logic** for a single
// computational graph node (`Node`) in the autodiff system.
//
// The purpose of `forward_eval_node()` is to *recompute* or *evaluate*
// a node’s output tensor based solely on its input nodes’ values,
// without using stored intermediate data.
//
// This is crucial for:
//    - Checkpoint recomputation (freeing and restoring activations),
//    - Lazy evaluation (on-demand computation),
//    - Debug visualization or forward-only inference.
//
// Additionally, the `checkpoint()` function here provides a user-facing API
// for marking specific nodes as checkpoints, integrating with the
// `checkpoint_impl` subsystem.
//
// Together, these functions enable **memory-efficient recomputation**
// during backward passes and safe graph traversal.
//


// -----------------------------------------------------------------------------
// forward_eval_node (shared_ptr<Node> version)
// -----------------------------------------------------------------------------

/*
 *  forward_eval_node():
 *  ---------------------
 *  Evaluates (or recomputes) the output tensor of a single computational node.
 *
 *  Parameters:
 *      - node : shared_ptr<Node> representing a node in the computational graph.
 *
 *  Returns:
 *      - A new Tensor that represents the computed output of this node,
 *        based on its operation type (`node->op`) and its input tensors.
 *
 *  Purpose:
 *      - This function allows recomputation of node outputs when they
 *        have been deleted or released during checkpointing.
 *      - It’s also used for lazy forward evaluation, debug visualization,
 *        or runtime validation of the computational graph.
 *
 *  Core logic:
 *      1️⃣  Validate that the node exists.
 *      2️⃣  Switch over the node’s operation (`Op` enum).
 *      3️⃣  Retrieve the node’s input tensors (`node->inputs[i]->value`).
 *      4️⃣  Perform the appropriate mathematical operation.
 *      5️⃣  Return the computed output tensor.
 *      6️⃣  If unsupported, throw a runtime error.
 */
#include <ad/autodiff/checkpoint.hpp>
Tensor forward_eval_node(const std::shared_ptr<Node> &node) {
    if (!node) throw std::runtime_error("forward_eval_node: null node");

    switch (node->op) {

        // ============================================================
        // Basic arithmetic operations
        // ============================================================
        case Op::Add: return node->inputs[0]->value + node->inputs[1]->value;
        case Op::Sub: return node->inputs[0]->value - node->inputs[1]->value;
        case Op::Mul: return node->inputs[0]->value * node->inputs[1]->value;

        // ============================================================
        // Matrix multiplication (dense layer or attention block)
        // ============================================================
        case Op::MatMul: {
            const Tensor &A = node->inputs[0]->value;
            const Tensor &B = node->inputs[1]->value;
            // Use the named async function for clarity
            return matmul(A, B);
        }

        // ============================================================
        // Unary elementwise activations
        // ============================================================
        case Op::Relu: {
            // Re-implement ReLU using the tensor library's ops
            auto& x = node->inputs[0]->value;
            return (x + OwnTensor::abs(x, ag::current_stream())) * 0.5f;
        }
        case Op::Linear: {
            const Tensor& input_X = node->inputs[0]->value;
            const Tensor& weight_W = node->inputs[1]->value; // Shape is [out, in]
            const Tensor& bias_b = node->inputs[2]->value;
            
            // This logic MUST match the forward logic in linear_nodeops
            return matmul(input_X, weight_W.t()) + bias_b;
        }
        // case Op::Sigmoid: {
        //     const Tensor &X = node->inputs[0]->value;
        //     return ag::sigmoid(X);
        // }

        case Op::GELU: {
            const Tensor &x = node->inputs[0]->value;
            // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            float k = std::sqrt(2.0f / 3.14159265358979323846f);
            auto x3 = OwnTensor::pow(x, 3.0f, ag::current_stream());
            auto inner = (x + x3 * 0.044715f) * k;
            auto tanh_inner = OwnTensor::tanh(inner, ag::current_stream());
            return x * 0.5f * (tanh_inner + 1.0f);
        }

        case Op::Tanh: {
            const Tensor &X = node->inputs[0]->value;
            return tanh(X);
        }
        case Op::Exp: {
            const Tensor &X = node->inputs[0]->value;
            return exp(X);
        }
        case Op::Log: {
            const Tensor &X = node->inputs[0]->value;
            return log(X);
        }
        

        // ============================================================
        // Leaf node (constants or inputs)
        // ============================================================
        /*
         * Op::Leaf:
         * ----------
         * Represents graph input nodes, constants, or parameters.
         * These do not require recomputation since their values
         * are provided externally or stored persistently.
         */
        case Op::Leaf:
            return node->value;

        // ============================================================
        // Default / fallback case
        // ============================================================
        /*
         * Handles cases where an operation type is not explicitly listed.
         * In some composite operations (like attention or layernorm),
         * intermediate tensors are temporarily stored in `node->tape`.
         *
         * If `tape` is not empty, it uses the last tensor in the tape
         * as a fallback recomputation result.
         */
        default:
            // if (!node->tape.empty()) {
            //     return *(node->tape.back());
            // }
            throw std::runtime_error(std::string("forward_eval_node: unsupported op for recompute: ") + op_name(node->op));
    }
}

// -----------------------------------------------------------------------------
// Adapter overload for raw pointer nodes
// -----------------------------------------------------------------------------

/*
 * forward_eval_node(Node*):
 * --------------------------
 *  Provides a lightweight wrapper around the main version of
 *  `forward_eval_node()` that takes a raw Node pointer instead of
 *  a shared_ptr.
 *
 *  This is used for internal integration with systems like
 *  checkpointing, which store and traverse raw Node* references.
 *
 *  Implementation ag::detail:
 *      - Wraps the raw Node* in a non-owning `shared_ptr<Node>`.
 *      - Uses a custom deleter `[](Node*){}` to prevent freeing.
 */
Tensor forward_eval_node(Node* node) {
    // Non-owning shared_ptr wrapper (no deletion)
    return forward_eval_node(std::shared_ptr<Node>(node, [](Node*){}));
}

// -----------------------------------------------------------------------------
// checkpoint() — Mark a node for checkpointing
// -----------------------------------------------------------------------------

/*
 * checkpoint():
 * --------------
 *  A user-facing function that marks a value (and its corresponding node)
 *  for checkpointing.
 *
 *  When a node is checkpointed:
 *      - Its intermediate activations may be freed to save memory.
 *      - During backpropagation, if its output is required,
 *        the system will recompute it using `forward_eval_node()`
 *        and its input dependencies.
 *
 *  Parameters:
 *      - v    : Value object wrapping the Node to be checkpointed.
 *      - opts : CheckpointOptions structure (default-initialized).
 *
 *  Returns:
 *      - The same Value `v` (allowing function chaining).
 *
 *  Internally, it calls:
 *      `checkpoint_impl::mark_node_checkpoint()`
 *  which performs the actual checkpoint marking and state saving.
 *
 *  Example usage:
 *      Value y = checkpoint(forward_pass(x));
 *      Tensor loss = mse(y, target);
 *      backward(loss);
 */


} // namespace ag
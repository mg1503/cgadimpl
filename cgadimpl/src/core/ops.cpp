// =====================
// file: cgadimpl/src/ops.cpp
// =====================
#include "ad/ops.hpp"
#include "ad/nodeops.hpp" // Include the new node-level declarations
#include "ad/inplace.hpp"

namespace ag {
    Value inplace_checkpoint(const Value& v) {
        if (!v.node) return v;
        ag::inplace::mark_inplace_checkpoint(v.node);
        return v;
    }

    Value add(const Value& a, const Value& b){ 
        return Value(detail::add_nodeops(a.node, b.node)); 
    }

    Value sub(const Value& a, const Value& b){ 
        
        return Value(detail::sub_nodeops(a.node, b.node)); 
    }



    Value mul(const Value& a, const Value& b){ 
        return Value(detail::mul_nodeops(a.node, b.node)); 
    }

    Value div(const Value& a, const Value& b){ 
        
        return Value(detail::div_nodeops(a.node, b.node)); 
    }

    Value sign(const Value& a, const Value& b){ 
        return Value(detail::sign_nodeops(a.node)); 
    }

    Value sinh(const Value& a, const Value& b){ 
        return Value(detail::sinh_nodeops(a.node)); 
    }

    Value cosh(const Value& a, const Value& b){ 
        return Value(detail::cosh_nodeops(a.node)); 
    }

    Value sin(const Value& a, const Value& b){ 
        return Value(detail::sin_nodeops(a.node)); 
    }

    Value cos(const Value& a, const Value& b){ 
        return Value(detail::cos_nodeops(a.node)); 
    }


    Value flomul(const Value& a, float b){ 
        return Value(detail::flomul_nodeops(a.node, b));
    }

    Value floadd(const Value& a, float b){ 
        return Value(detail::floadd_nodeops(b, a.node));
    }

     Value flodiv(const Value& a, float b){ 
        return Value(detail::flodiv_nodeops(b, a.node));
    }

    Value relu(const Value& x){ 
      
        return Value(detail::relu_nodeops(x.node));
    }

    Value reci(const Value& x){ 
      
        return Value(detail::reci_nodeops(x.node));
    }

    Value cos(const Value& x){ 
      
        return Value(detail::cos_nodeops(x.node));
    }

    Value sinh(const Value& x){ 
      
        return Value(detail::sinh_nodeops(x.node));
    }

    Value cosh(const Value& x){ 
      
        return Value(detail::cosh_nodeops(x.node));
    }




    Value sigatt(const Value& a, const Value& b, const Value& c, const Value& d){ 
    return Value(detail::sigatt_nodeops(a.node, b.node, c.node, d.node));
    }

    Value linear(const Value& a, const Value& b, const Value& c){ 
        return Value(detail::linear_nodeops(a.node, b.node, c.node)); 
    }


        Value moewe(const Value& x, const Value& w, const Value& b){ 
        return Value(detail::moewe_nodeops(x.node, w.node, b.node));
    }



    Value matmul(const Value& a, const Value& b){ 
         return Value(detail::matmul_nodeops(a.node, b.node)); 
    }

    Value fmab(const Value& a, const Value& b, const Value& c){ 
        return Value(detail::fmab_nodeops(a.node, b.node, c.node)); 
    }


    Value attention(const Value& a, const Value& b, const Value& c, const Value& d){ 
    return Value(detail::attention_nodeops(a.node, b.node, c.node, d.node));
    }


    Value alibiatt(const Value& a, const Value& b, const Value& c, const Value& d, float m) { 
    return Value(detail::alibiatt_nodeops(a.node, b.node, c.node, d.node, m));
}



    Value swiglu(const Value& x, const Value& a, const Value& b, const Value& c, const Value& d){ 
    return Value(detail::swiglu_nodeops(x.node, a.node, b.node, c.node, d.node));
    }


    Value sum(const Value& x){ 
        return Value(detail::sum_nodeops(x.node));
    }

    Value transpose(const Value& x){ 
        return Value(detail::transpose_nodeops(x.node));
    }

    Value exp(const Value& x){ 
        return Value(detail::exp_nodeops(x.node));
    }
    
    Value log(const Value& x){ 
        return Value(detail::log_nodeops(x.node));
    }


    Value mish(const Value& x){ 
        return Value(detail::mish_nodeops(x.node));
    }
    
    Value tanh(const Value& x){ 
        return Value(detail::tanh_nodeops(x.node));
    }

    
    Value reluatt(const Value& a, const Value& b, const Value& c, const Value& d){ 
    return Value(detail::reluatt_nodeops(a.node, b.node, c.node, d.node));
    }


  


 

    Value sigmoid(const Value& x){ 
        return Value(detail::sigmoid_nodeops(x.node));
    }
    
    Value softplus(const Value& x){ 
        return Value(detail::softplus_nodeops(x.node));
    }

    Value gaus(const Value& x){ 
        return Value(detail::gaus_nodeops(x.node));
    }
    
    Value gelu(const Value& x){ 
        return Value(detail::gelu_nodeops(x.node));
    }



    Value gcu(const Value& x){ 
        return Value(detail::gcu_nodeops(x.node));
    }
    
    Value silu(const Value& x){ 
        return Value(detail::silu_nodeops(x.node));
    }

    Value parcon(const Value& x){ 
        return Value(detail::parcon_nodeops(x.node));
    }

    Value lisht(const Value& x){ 
        return Value(detail::lisht_nodeops(x.node));
    }
    
    Value leaky_relu(const Value& x, float alpha){ 
        return Value(detail::leaky_relu_nodeops(x.node, alpha));
    }


    Value rowsum(const Value& x){ 
        return Value(detail::rowsum_nodeops(x.node));
    }
    
    Value rowmax(const Value& x){ 
        return Value(detail::rowmax_nodeops(x.node));
    }

    Value rms(const Value& x){ 
return Value(detail::rms_nodeops(x.node));
    }

    Value realrms(const Value& x, float g){ 
return Value(detail::realrms_nodeops(x.node, g));
    }

    Value laynor(const Value& x){ 
        return Value(detail::laynor_nodeops(x.node));
    }

    Value relaynor(const Value& x, float b, float g){ 
        return Value(detail::relaynor_nodeops(x.node, b, g));
    }
    
    Value mean_all(const Value& x){ 
        return Value(detail::mean_all_nodeops(x.node));
    }

    Value dyntanh(const Value& x, float a, float b, float g){ 
        return Value(detail::dyntanh_nodeops(x.node, a, b, g));
    }
    
    Value softmax_row(const Value& z){ 
        return Value(detail::softmax_row_nodeops(z.node));
    }
    
    Value logsumexp_row(const Value& z){ 
        return Value(detail::logsumexp_row_nodeops(z.node));
    }


    Value mambassm(const Value& z, const Value& a, const Value& b, const Value& c, const Value& d){ 

        return Value(detail::mambassm_nodeops(z.node, a.node, b.node, c.node, d.node));

        
    }


    Value cross_entropy_with_logits(const Value& logits, const Value& onehot){
    // Stable CE = mean( -sum(onehot * _nodeops(logits - logsumexp_row_nodeops(logits))) )
        return Value(detail::cross_entropy_with_logits_nodeops(logits.node, onehot.node));
    }


    Value kldivergence(const Value& logits, const Value& onehot){
        return Value(detail::kldivergence_nodeops(logits.node, onehot.node));
    }

    Value mse_loss(const Value& pred, const Value& target) {
    return Value(detail::mse_loss_nodeops(pred.node, target.node));
}


    Value mae_loss(const Value& pred, const Value& target) {
    return Value(detail::mae_loss_nodeops(pred.node, target.node));
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
Tensor forward_eval_node(const std::shared_ptr<Node> &node) {
    if (!node) throw std::runtime_error("forward_eval_node: null node");

    switch (node->op) {

        // ============================================================
        // Basic arithmetic operations
        // ============================================================
        case Op::Add: {
            const Tensor &A = node->inputs[0]->value;
            const Tensor &B = node->inputs[1]->value;
            return A + B;
        }
        case Op::Sub: {
            const Tensor &A = node->inputs[0]->value;
            const Tensor &B = node->inputs[1]->value;
            return A - B;
        }
        case Op::Mul: {
            const Tensor &A = node->inputs[0]->value;
            const Tensor &B = node->inputs[1]->value;
            return A * B;
        }

        // ============================================================
        // Matrix multiplication (dense layer or attention block)
        // ============================================================
        case Op::MatMul: {
            const Tensor &A = node->inputs[0]->value;
            const Tensor &B = node->inputs[1]->value;
            return Tensor::matmul(A, B);
        }

        // ============================================================
        // Unary elementwise activations
        // ============================================================
        case Op::Relu: {
            const Tensor &X = node->inputs[0]->value;
            return Tensor::relu(X);
        }
        case Op::Sigmoid: {
            const Tensor &X = node->inputs[0]->value;
            return Tensor::sigmoid(X);
        }
        case Op::Tanh: {
            const Tensor &X = node->inputs[0]->value;
            return Tensor::tanh(X);
        }
        case Op::Exp: {
            const Tensor &X = node->inputs[0]->value;
            return Tensor::exp(X);
        }
        case Op::Log: {
            const Tensor &X = node->inputs[0]->value;
            return Tensor::log(X);
        }

        // ============================================================
        // Complex operation: AlibiAttention
        // ============================================================
        /*
         * AlibiAttention:
         * ---------------
         * This is a specialized attention mechanism variant that adds
         * a learned or deterministic bias (ALIBI) to the attention logits.
         *
         * Steps:
         *    1. Compute queries (q), keys (k), and values (v) via matmul.
         *    2. Compute scaled dot-product attention scores.
         *    3. Apply ALIBI positional bias.
         *    4. Compute softmax over the attention weights.
         *    5. Multiply attention weights with the values to get the output.
         */
        case Op::AlibiAttention: {
            const Tensor &a = node->inputs[0]->value;
            const Tensor &b = node->inputs[1]->value;
            const Tensor &c = node->inputs[2]->value;
            const Tensor &d = node->inputs[3]->value;

            // Step 1: compute projections
            Tensor q = Tensor::matmul(a, b);
            Tensor k = Tensor::matmul(a, c);
            Tensor v = Tensor::matmul(a, d);

            // Step 2: scaled dot-product attention
            Tensor logits = Tensor::matmul(q, Tensor::transpose(k) * (1.f / sqrt(float(k.cols()))));

            // Step 3: add ALIBI bias (creates a position-dependent attention slope)
            Tensor bias = Tensor::alibi(logits.rows(), logits.cols(), /*m*/128);
            Tensor g = logits + bias;

            // Step 4: softmax normalization over rows
            Tensor s = Tensor::softmax_row(g);

            // Step 5: output = attention weights × values
            Tensor y = Tensor::matmul(s, v);
            return y;
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
            if (!node->tape.empty()) {
                return *(node->tape.back());
            }
            throw std::runtime_error("forward_eval_node: unsupported op for recompute");
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
 *  Implementation detail:
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
Value checkpoint(const Value &v, const CheckpointOptions &opts) {
    if (!v.node) return v;
    ag::checkpoint_impl::mark_node_checkpoint(v.node, opts);
    return v;
}

} // namespace ag
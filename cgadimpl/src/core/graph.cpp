// =====================
// file: cgadimpl/src/graph.cpp
// =====================
#include "ad/graph.hpp"
#include <unordered_set>
#include <functional>
#include <cassert>



namespace ag {

// --- Node Implementation ---
// Node::Node() = default; 
Node::Node(const Tensor& v, Op op_, const char* nm) 
    : op(op_), 
      value(v), 
      // Initialize `grad` with an empty but valid Tensor using TensorOptions.
      grad(), 
      debug_name(nm) 
{
    if (v.requires_grad()) {
        // CORRECT WAY:
        // 1. Create a TensorOptions object with the correct properties.
        // TensorOptions opts = TensorOptions()
        //                         .with_dtype(v.dtype())
        //                         .with_device(v.device());
        
        // 2. Call the 'zeros' factory with the correct signature (shape, opts).
        grad = OwnTensor::Tensor::zeros(v.shape(), ag::options(v));
    }/*else {
        // If no grad is required, grad can be an empty tensor.
        grad = Tensor(Shape{}, TensorOptions().with_dtype(v.dtype()).with_device(v.device()));
    }*/
}

// --- Value Implementation ---
// ADDED: Implement the Value helper functions
Tensor& Value::val() { return node->value; }
const Tensor& Value::val() const { return node->value; }
Tensor& Value::grad() { return node->grad; }
const Tensor& Value::grad() const { return node->grad; }
Value::Value() = default;
Value::Value(std::shared_ptr<Node> n) : node(std::move(n)) {}

// NEW: Implementation for the real shape()
const std::vector<int64_t>& Value::shape() const {
    return node->value.shape().dims;
}
// 2d helper
std::pair<int, int> Value::shape_2d() const {
    const auto& dims = node->value.shape().dims;
    if (dims.size() == 0) return {0, 0};
    if (dims.size() == 1) return {1, static_cast<int>(dims[0])};
    // For 2D or more, return the first two dimensions.
    return {static_cast<int>(dims[0]), static_cast<int>(dims[1])};
}

// // --- Factory Implementation ---
// Value make_tensor(const Tensor& v, const char* name) {
//     return Value(std::make_shared<Node>(v, Op::Leaf, name));
// }

// --- Graph Traversal ---
std::vector<Node*> topo_from(Node* root){
    std::vector<Node*> order; order.reserve(256);
    std::unordered_set<Node*> vis; vis.reserve(256);
    std::function<void(Node*)> dfs = [&](Node* n){ if(!n || vis.count(n)) return; vis.insert(n); for(auto& p : n->inputs) dfs(p.get()); order.push_back(n); };
    dfs(root);
    return order; // parents before child
}

} // namespace ag
// ===================================================================
// JIT COMPILER IMPLEMENTATION -- REMAINS TEMPORARILY DISABLED
//     Deleted broadcast_to and mul_scalar: These helper functions are no longer needed. The new OwnTensor library overloads operators like + and * to handle broadcasting automatically. This simplifies the code.
//     Rewrote apply(): This is the most significant change.
//         Binary Operators: Changed A + B to *a[0] + *a[1]. The OwnTensor operators will handle everything.
//         Unary Operators: Changed Tensor::relu(*a[0]) to the free function OwnTensor::relu(*a[0]). This pattern must be applied to all unary functions.
//         Missing Functions: I've commented out functions like softplus, silu, and softmax_row. The OwnTensor library, as provided, does not have these functions. To fully enable the JIT, you would need to either:
//             Add these functions to the tensor library.
//             Implement them manually inside the apply function using more basic ops (exp, log, /, etc.).
//         Reductions: The reduction calls were updated to the new API, like OwnTensor::reduce_sum(*a[0]). For RowSum, we now call reduce_sum with the correct axis ({1}) and keepdim=true.
//     Corrected run():
//         The slots pre-allocation loop was already fixed by you in the previous step. It's correct.
//         The tmp variable for literals needed to be initialized with the new default constructor pattern, which I've fixed.
// Next Steps:
//     Replace the entire Compiled::Impl struct in cgadimpl/src/graph.cpp with the rewritten version above.
//     Decide what to do about the missing functions (like silu, softmax_row). For now, leaving them commented out is the safest option to get the code to compile.
// ===================================================================  
#include <unordered_map>
#include <variant>

namespace ag::jit {
// ===================================================================
// JIT Compiler Signature (Rewritten for N-Dimensional Tensors)
// ===================================================================
struct Signature {
    // --- CHANGE #1: Store the full, N-dimensional shape ---
    // We now store a vector of vectors, where each inner vector is a full shape.
    std::vector<std::vector<int64_t>> in_shapes;
    std::vector<std::vector<int64_t>> param_shapes;

    bool matches(const std::vector<Tensor*>& inputs,
                 const std::vector<Tensor*>& params) const {
        if (inputs.size() != in_shapes.size() || params.size() != param_shapes.size()) {
            return false;
        }

        // --- CHANGE #2: Update the comparison logic ---
        for (size_t i = 0; i < inputs.size(); ++i) {
            // Get the real shape (a vector of int64_t) from the new Tensor class.
            const auto& s = inputs[i]->shape().dims;
            
            // The comparison now works correctly because both `s` and `in_shapes[i]`
            // are of the same type: std::vector<int64_t>.
            if (s != in_shapes[i]) {
                return false;
            }
        }
        for (size_t i = 0; i < params.size(); ++i) {
            // Do the same for params.
            const auto& s = params[i]->shape().dims;
            if (s != param_shapes[i]) {
                return false;
            }
        }
        return true;
    }
};
// Arg sources for a Step
struct ArgInput  { int idx; };   // external input[i]
struct ArgParam  { int idx; };   // external param[i]
struct ArgSlot   { int slot; };  // prior computed slot
struct ArgLit    { Tensor t{OwnTensor::Shape{}, OwnTensor::Dtype::Float32}; };  // embedded literal

using Arg = std::variant<ArgInput,ArgParam,ArgSlot,ArgLit>;

struct Step {
    Op op;
    std::vector<Arg> args;
    int out_slot{};                 // where to write result
    std::vector<int64_t> out_shape{}; // rows,cols
};

struct Plan {
    Signature sig;
    std::vector<Step> steps;
    int num_slots{0};
    int out_slot{-1};
};

struct Compiled::Impl {
    Plan plan;

    // --- helpers for replay ---
    static const Tensor& as_ref(const Arg& a,
                                const std::vector<Tensor*>& inputs,
                                const std::vector<Tensor*>& params,
                                const std::vector<Tensor>& slots,
                                Tensor& tmp) {
        if (std::holds_alternative<ArgInput>(a))  return *inputs[std::get<ArgInput>(a).idx];
        if (std::holds_alternative<ArgParam>(a))  return *params[std::get<ArgParam>(a).idx];
        if (std::holds_alternative<ArgSlot>(a))   return slots[std::get<ArgSlot>(a).slot];
        // literal: copy into tmp to return a ref
        const Tensor& lit = std::get<ArgLit>(a).t;
        tmp = lit;
        return tmp;
    }


    static Tensor apply(Op op, const std::vector<const Tensor*>& a) {
        // a.size() equals op_arity(op), except literals we materialized as tensors
        switch(op){
            case Op::Add:        return *a[0] + *a[1];
            case Op::Sub:        return *a[0] - *a[1];
            case Op::Mul:        return *a[0] * *a[1];

            // Unary operators now use the free functions from the OwnTensor namespace.
            case Op::Transpose:  return a[0]->transpose(-2, -1);
            // case Op::Relu:       return OwnTensor::relu(*a[0]);
            case Op::Exp:        return OwnTensor::exp(*a[0]);
            case Op::Log:        return OwnTensor::log(*a[0]);
            case Op::Tanh:       return OwnTensor::tanh(*a[0]);
            // case Op::Sigmoid:    return Tensor::sigmoid(*a[0]);
            // case Op::Softplus:   return Tensor::softplus(*a[0]);
            // case Op::SiLU:       return Tensor::silu(*a[0]);
            // case Op::GELU:       return Tensor::gelu(*a[0]);
            // case Op::LeakyRelu: {
            //     // The new API would likely be a free function.
            //     // Let's assume it's OwnTensor::leaky_relu(tensor, alpha).
            //     // The alpha value is stored in the second input tensor.
            //     float alpha = a[1]->data<float>()[0];
            //     return OwnTensor::leaky_relu(*a[0], alpha);
            // }

            case Op::MatMul:     return OwnTensor::matmul(*a[0], *a[1]);

            // Reductions need to be updated to the new API
            case Op::Sum:        return OwnTensor::reduce_sum(*a[0]);
            case Op::RowSum:     return OwnTensor::reduce_sum(*a[0], {1}, true); // reduce axis 1, keepdim
            case Op::RowMax:     return OwnTensor::reduce_max(*a[0], {1}, true); // reduce axis 1, keepdim
            
            case Op::MeanAll:    return OwnTensor::reduce_mean(*a[0]);

            // case Op::SoftmaxRow: return Tensor::softmax_row(*a[0]);
            // case Op::LogSumExpRow: return Tensor::logsumexp_row(*a[0]);
            // case Op::CeWithLogits: {
            //     // CE = -mean( sum( Y * (Z - lse(Z)), axis=1 ) )
            //     const Tensor& Z = *a[0];
            //     const Tensor& Y = *a[1];
            //     Tensor lse = Tensor::logsumexp_row(Z);           // [B,1]
            //     // broadcast lse to [B,C]
            //     Tensor L = broadcast_to(lse, Z.rows(), Z.cols());
            //     Tensor term = Y * (Z - L);                        // [B,C]
            //     Tensor rs = Tensor::row_sum(term);                // [B,1]
            //     Tensor s = Tensor::sum_all(rs);                   // [1,1]
            //     float invB = -1.f / float(Z.rows());
            //     return mul_scalar(s, invB);
            // }
            case Op::Leaf: default: {
                // Shouldn't get called for Leaf
                assert(false && "apply(): unexpected op");
                return *a[0];
            }
        }
    }

    bool run(const std::vector<Tensor*>& inputs,
             const std::vector<Tensor*>& params,
             Tensor& out) const {
        if (!plan.sig.matches(inputs, params)) return false;

        std::vector<Tensor> slots(plan.num_slots);
        
        // This loop is now correct thanks to your previous fix.
        for (const Step& st : plan.steps) {
            if (st.out_slot >= 0) {
                slots[st.out_slot] = Tensor(OwnTensor::Shape{st.out_shape}, false);
            }
        }

        // Execute
        for (const Step& st : plan.steps) {
            std::vector<const Tensor*> args; args.reserve(st.args.size());
            
            // Corrected default constructor for tmp
            Tensor tmp{OwnTensor::Shape{}, OwnTensor::TensorOptions{}}; 
            
            std::vector<Tensor> tmp_keep; tmp_keep.reserve(st.args.size());
            for (const Arg& a : st.args) {
                if (std::holds_alternative<ArgLit>(a)) {
                    tmp_keep.emplace_back(std::get<ArgLit>(a).t);
                    args.push_back(&tmp_keep.back());
                } else {
                    // as_ref is fine, no changes needed here.
                    args.push_back(&as_ref(a, inputs, params, slots, tmp));
                }
            }
            Tensor y = apply(st.op, args);
            slots[st.out_slot] = std::move(y);
        }

        out = slots[plan.out_slot];
        return true;
    }
};
static bool is_in(const std::unordered_map<Node*,int>& m, Node* n){ return m.find(n)!=m.end(); }

Compiled compile(const Value& output,
                 const std::vector<Value>& inputs,
                 const std::vector<Value>& params,
                 const CompileOptions&) {
    // Map externals (no changes needed here)
    std::unordered_map<Node*,int> in_ix, par_ix;
    // ...

    // Build plan
    Plan plan;

    // --- CHANGE #1: Populate the Signature with the correct N-D shape vector ---
    plan.sig.in_shapes.reserve(inputs.size());
    for (const auto& v: inputs) {
        // v.val() is a Tensor. .shape() is its member function returning a vector.
        plan.sig.in_shapes.push_back(v.shape()); 
    }
    plan.sig.param_shapes.reserve(params.size());
    for (const auto& v: params) {
        plan.sig.param_shapes.push_back(v.shape());
    }

    auto order = topo_from(output.node.get());
    std::unordered_map<Node*,int> slot_of;
    slot_of.reserve(order.size());

    for (Node* n : order) {
        if (n->op == Op::Leaf) {
            continue;
        }
        Step st;
        st.op = n->op;

        // --- CHANGE #2: Populate the Step with the correct N-D shape vector ---
        st.out_shape = n->shape(); // Use the new .shape() helper on the Node

        st.out_slot = plan.num_slots++;
        slot_of[n] = st.out_slot;

        // ... (Gather args logic is correct and needs no changes) ...
         st.args.reserve(n->inputs.size());
        for (auto& pin : n->inputs) {
            Node* p = pin.get();
            if (p->op == Op::Leaf) {
                if (is_in(in_ix, p))        st.args.push_back(ArgInput{ in_ix[p] });
                else if (is_in(par_ix, p))  st.args.push_back(ArgParam{ par_ix[p] });
                else                        st.args.push_back(ArgLit{ p->value }); // embedded literal leaf
            } else {
                // computed parent
                st.args.push_back(ArgSlot{ slot_of.at(p) });
            }
        }


        plan.steps.push_back(std::move(st));
    }

    // Final slot
    plan.out_slot = slot_of.at(output.node.get());

    Compiled c;
    c.p = std::make_shared<Compiled::Impl>();
    c.p->plan = std::move(plan);
    return c;
}

bool Compiled::run(const std::vector<Tensor*>& inputs,
                   const std::vector<Tensor*>& params,
                   Tensor& out) const {
    return p->run(inputs, params, out);
}

} // namespace ag::jit

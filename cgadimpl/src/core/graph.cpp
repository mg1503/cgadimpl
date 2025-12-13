// =====================
// file: cgadimpl/src/graph.cpp
// =====================
#include "ad/graph.hpp"
#include "ad/mlir_emitter.hpp"
#include <sstream>
#include <unordered_set>
#include <functional>
#include <cassert>
#include <iostream> // Added for printing


namespace ag {

// --- Node Implementation ---
// Node::Node() = default; 
Node::Node(const Tensor& v, Op op_, bool req_grad, const char* nm) 
    : op(op_), 
      value(v),
      requires_grad_flag_(req_grad),
      debug_name(nm) 
{
    if (requires_grad_flag_) {
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

// --- Internal implementation for graph traversal ---
static std::vector<Node*> build_topo_order_impl(Node* root) {
    std::vector<Node*> order; order.reserve(256);
    std::unordered_set<Node*> vis; vis.reserve(256);
    std::function<void(Node*)> dfs = [&](Node* n){ if(!n || vis.count(n)) return; vis.insert(n); for(auto& p : n->inputs) dfs(p.get()); order.push_back(n); };
    dfs(root);
    return order; // parents before child
}

// A cache for memoizing topological sorts of graphs.
static std::unordered_map<Node*, std::vector<Node*>> topo_cache;

// --- Graph Traversal ---
std::vector<Node*> topo_from(Node* root){
    // Check if the graph order is already cached
    auto it = topo_cache.find(root);
    if (it != topo_cache.end()) {
        return it->second; // Cache hit
    }

    // Cache miss: build the order, cache it, and return it
    std::cout << "--- Building and Caching Computational Graph ---" << std::endl;
    std::vector<Node*> order = build_topo_order_impl(root);
    topo_cache[root] = order;
    return order;
}

} // namespace ag
// ===================================================================
// JIT COMPILER IMPLEMENTATION 
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
#include <ad/runtime.hpp>

namespace ag::jit {
// ===================================================================
// JIT Compiler Signature (Rewritten for N-Dimensional Tensors)
// ===================================================================
struct TensorMetadata {
    std::vector<int64_t> shape;
    Dtype dtype;
    DeviceIndex device;
};

struct Signature {
    std::vector<TensorMetadata> in_meta;
    std::vector<TensorMetadata> param_meta;

    bool matches(const std::vector<Tensor*>& inputs,
                 const std::vector<Tensor*>& params) const {
        if (inputs.size() != in_meta.size() || params.size() != param_meta.size()) {
            return false;
        }

        for (size_t i = 0; i < inputs.size(); ++i) {
            if (inputs[i]->shape().dims != in_meta[i].shape ||
                inputs[i]->dtype() != in_meta[i].dtype ||
                inputs[i]->device().device != in_meta[i].device.device ||
                inputs[i]->device().index != in_meta[i].device.index) {
                return false;
            }
        }
        for (size_t i = 0; i < params.size(); ++i) {
            if (params[i]->shape().dims != param_meta[i].shape ||
                params[i]->dtype() != param_meta[i].dtype ||
                params[i]->device().device != param_meta[i].device.device ||
                params[i]->device().index != param_meta[i].device.index) {
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
    int out_slot{};
    TensorMetadata out_meta;
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
            case Op::Relu:     { cudaStream_t stream = (cudaStream_t)ag::current_stream(); return (*a[0] + OwnTensor::abs(*a[0], stream)) * 0.5f;}
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
            case Op::Sum: return OwnTensor::reduce_sum(*a[0]);
            case Op::RowSum: return OwnTensor::reduce_sum(*a[0], {1}, true);
            case Op::RowMax: return OwnTensor::reduce_max(*a[0], {1}, true);
            case Op::MeanAll: return OwnTensor::reduce_mean(*a[0]);

            // case Op::Abs:       { cudaStream_t stream = (cudaStream_t)ag::current_stream(); return OwnTensor::abs(*a[0], stream); }

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
                slots[st.out_slot] = Tensor(OwnTensor::Shape{st.out_meta.shape}, st.out_meta.dtype, st.out_meta.device, false);
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
    // Map externals
    std::unordered_map<Node*,int> in_ix, par_ix;
    for (size_t i = 0; i < inputs.size(); ++i) {
        in_ix[inputs[i].node.get()] = static_cast<int>(i);
    }
    for (size_t i = 0; i < params.size(); ++i) {
        par_ix[params[i].node.get()] = static_cast<int>(i);
    }

    // Build plan
    Plan plan;

    // Populate the Signature with full metadata
    plan.sig.in_meta.reserve(inputs.size());
    for (const auto& v: inputs) {
        plan.sig.in_meta.push_back({v.shape(), v.val().dtype(), v.val().device()});
    }
    plan.sig.param_meta.reserve(params.size());
    for (const auto& v: params) {
        plan.sig.param_meta.push_back({v.shape(), v.val().dtype(), v.val().device()});
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

        // Populate the Step with full output metadata
        st.out_meta = {n->shape(), n->value.dtype(), n->value.device()};

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
    // std::cout<<"\n-------JIT Execution Plan--------------\n";

    // std::cout<<"signature:\n";
    // for(size_t i=0; i < plan.sig.in_shapes.size(); ++i){
    //     std::cout<<" Input " << i << " Shape: [";
    //     for (size_t j = 0; j < plan.sig.in_shapes[i].size(); ++j){
    //         std::cout<<plan.sig.in_shapes[i][j] << (j == plan.sig.in_shapes[i].size() - 1 ? "":", ");
    //     }
    //     std::cout << "]\n";
    // }
    // for (size_t i = 0; i < plan.sig.param_shapes.size(); ++i){
    //     std::cout<<" Param " << i << " Shape: [";
    //     for(size_t j = 0; j < plan.sig.param_shapes[i].size(); ++j){
    //         std::cout<<plan.sig.param_shapes[i][j] << (j == plan.sig.param_shapes[i].size() - 1 ? "":", ");
    //     }
    //     std::cout <<"]\n";
    // }
    // std::cout<<"num slots:"<<plan.num_slots<<'\n';
    // std::cout<<"out_slot"<<plan.out_slot<<'\n';

// Compiled c = compile(output, inputs, params);
//     const Plan& plan = c.p->plan; // Assuming c.p is a shared_ptr to Impl
 
    // for (size_t i = 0; i < plan.steps.size(); ++i) {
    // const Step& st = plan.steps[i];
    // std::cout << "Step " << i << ": Op = " << op_name(st.op)
    //           << ", OutSlot = " << st.out_slot
    //           << ", OutShape = (" << st.out_shape[i] <<"\n";
    // std::cout << "  Args: ";
    // for (const auto& arg : st.args) {
    //     if (std::holds_alternative<ArgInput>(arg))
    //         std::cout << "[Input idx " << std::get<ArgInput>(arg).idx << "] ";
    //     else if (std::holds_alternative<ArgParam>(arg))
    //         std::cout << "[Param idx " << std::get<ArgParam>(arg).idx << "] ";
    //     else if (std::holds_alternative<ArgSlot>(arg))
    //         std::cout << "[Slot " << std::get<ArgSlot>(arg).slot << "] ";
    //     else if (std::holds_alternative<ArgLit>(arg))
    //         std::cout << "[Literal] ";
    // }
    // for(size_t i=0; i < plan.sig.in_shapes.size(); ++i){
    //     std::cout<<" Input " << i << " Shape: [";
    //     for (size_t j = 0; j < plan.sig.in_shapes[i].size(); ++j){
    //         std::cout<<plan.sig.in_shapes[i][j] << (j == plan.sig.in_shapes[i].size() - 1 ? "":", ");
    //     }
    //     std::cout << "]\n";
    // }
    // for (size_t i = 0; i < plan.sig.param_shapes.size(); ++i){
    //     std::cout<<" Param " << i << " Shape: [";
    //     for(size_t j = 0; j < plan.sig.param_shapes[i].size(); ++j){
    //         std::cout<<plan.sig.param_shapes[i][j] << (j == plan.sig.param_shapes[i].size() - 1 ? "":", ");
    //     }
    //     std::cout <<"]\n";
    // }
        // Helper to print shape
        auto print_shape_vec = [](const std::vector<int64_t>& shape) {
            std::cout << "[";
            for (size_t k = 0; k < shape.size(); ++k) {
                std::cout << shape[k] << (k == shape.size() - 1 ? "" : ", ");
            }
            std::cout << "]";
        };

        // Helper to print dtype
        auto get_dtype_str = [](Dtype dt) -> std::string {
            switch(dt) {
                case Dtype::Float32: return "fp32";
                case Dtype::Float16: return "fp16";
                case Dtype::Bfloat16: return "bf16";
                case Dtype::Int32: return "i32";
                case Dtype::Int64: return "i64";
                default: return "unknown";
            }
        };

        for (size_t i = 0; i < plan.steps.size(); ++i) {
            const auto& st = plan.steps[i];
            std::cout << "Step " << i << ": slot[" << st.out_slot << "] = " << op_name(st.op) << "(";
     
            // Print arguments
            for (size_t j = 0; j < st.args.size(); ++j) {
                std::visit([&](auto&& arg) {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, ArgInput>) {
                        const auto& m = plan.sig.in_meta[arg.idx];
                        std::cout << get_dtype_str(m.dtype);
                        print_shape_vec(m.shape);
                    } else if constexpr (std::is_same_v<T, ArgParam>) {
                        const auto& m = plan.sig.param_meta[arg.idx];
                        std::cout << get_dtype_str(m.dtype);
                        print_shape_vec(m.shape);
                    } else if constexpr (std::is_same_v<T, ArgSlot>) {
                        // Find the step that produced this slot to get its shape
                        bool found = false;
                        for (size_t prev_i = 0; prev_i < i; ++prev_i) {
                            if (plan.steps[prev_i].out_slot == arg.slot) {
                                const auto& m = plan.steps[prev_i].out_meta;
                                std::cout << get_dtype_str(m.dtype);
                                print_shape_vec(m.shape);
                                found = true;
                                break;
                            }
                        }
                        if (!found) std::cout << "slot[" << arg.slot << "]";
                    } else if constexpr (std::is_same_v<T, ArgLit>) {
                        std::cout << get_dtype_str(arg.t.dtype());
                        print_shape_vec(arg.t.shape().dims);
                    }
                }, st.args[j]);
    
                if (j < st.args.size() - 1) std::cout << ", ";
            }
    
            std::cout << ") -> shape ";
            print_shape_vec(st.out_meta.shape);
            
            std::cout << " -> Device [" << (st.out_meta.device.is_cpu() ? "CPU" : "CUDA") << "]\n";
        }
    std::cout << "\\n";
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

std::string Compiled::toMLIR(const std::string& function_name) const {
    return mlir_emit::emitNovaDialect(p->plan, function_name);
}

// ===================================================================
// MLIR Nova Dialect Emission Implementation
// ===================================================================
namespace mlir_emit {

std::string emitNovaDialect(const Plan& plan, const std::string& function_name) {
    std::ostringstream mlir;
    
    // Assume Float32 for now - in a real implementation, you'd track actual dtypes
    const OwnTensor::Dtype default_dtype = OwnTensor::Dtype::Float32;
    
    // Build function signature
    mlir << "func.func @" << function_name << "(";
    
    // Emit input arguments
    for (size_t i = 0; i < plan.sig.in_meta.size(); ++i) {
        if (i > 0) mlir << ", ";
        mlir << "'%'arg" << i << ": " 
             << ag::jit::mlir_emit::shapeToMLIRType(plan.sig.in_meta[i].shape, plan.sig.in_meta[i].dtype);
    }
    
    // Emit parameter arguments
    for (size_t i = 0; i < plan.sig.param_meta.size(); ++i) {
        if (plan.sig.in_meta.size() > 0 || i > 0) mlir << ", ";
        mlir << "'%'arg" << (plan.sig.in_meta.size() + i) << ": "
             << ag::jit::mlir_emit::shapeToMLIRType(plan.sig.param_meta[i].shape, plan.sig.param_meta[i].dtype);
    }
    
    mlir << ") -> ";
    
    // Emit return type (output shape from final slot)
    if (plan.out_slot >= 0 && plan.out_slot < plan.num_slots) {
        // Find the step that writes to out_slot to get its metadata
        std::vector<int64_t> out_shape;
        Dtype out_dtype = default_dtype;
        for (const auto& step : plan.steps) {
            if (step.out_slot == plan.out_slot) {
                out_shape = step.out_meta.shape;
                out_dtype = step.out_meta.dtype;
                break;
            }
        }
        mlir << ag::jit::mlir_emit::shapeToMLIRType(out_shape, out_dtype);
    } else {
        mlir << "tensor<f32>"; // fallback
    }
    
    mlir << " {\n";
    
    // Track SSA value names for slots
    std::unordered_map<int, std::string> slot_to_ssa;
    int ssa_counter = 0;
    
    // Emit operations
    for (const auto& step : plan.steps) {
        mlir << "  ";
        
        // Emit result SSA value
        std::string result_ssa = "%" + std::to_string(ssa_counter++);
        slot_to_ssa[step.out_slot] = result_ssa;
        mlir << result_ssa << " = ";
        
        // Emit operation
        if (isReductionOp(step.op)) {
            // Special handling for reduction operations
            mlir << "nova.reduce<" << getReductionKind(step.op) << "> ";
            
            // Emit input operand
            assert(step.args.size() >= 1);
            const Arg& arg = step.args[0];
            
            if (std::holds_alternative<ArgInput>(arg)) {
                mlir << "'%'arg" << std::get<ArgInput>(arg).idx;
            } else if (std::holds_alternative<ArgParam>(arg)) {
                int param_idx = std::get<ArgParam>(arg).idx + plan.sig.in_meta.size();
                mlir << "'%'arg" << param_idx;
            } else if (std::holds_alternative<ArgSlot>(arg)) {
                mlir << slot_to_ssa.at(std::get<ArgSlot>(arg).slot);
            }
            
            // Add dimension attribute if needed
            if (needsDimensionAttr(step.op)) {
                mlir << " dimension=[1] keepdims=true";
            }
            
            mlir << " : " << ag::jit::mlir_emit::shapeToMLIRType(step.out_meta.shape, step.out_meta.dtype);
            
        } else {
            // Regular operations
            mlir << opToNovaOp(step.op) << " ";
            
            // Emit operands
            for (size_t i = 0; i < step.args.size(); ++i) {
                if (i > 0) mlir << ", ";
                
                const Arg& arg = step.args[i];
                
                if (std::holds_alternative<ArgInput>(arg)) {
                    mlir << "'%'arg" << std::get<ArgInput>(arg).idx;
                } else if (std::holds_alternative<ArgParam>(arg)) {
                    int param_idx = std::get<ArgParam>(arg).idx + plan.sig.in_meta.size();
                    mlir << "'%'arg" << param_idx;
                } else if (std::holds_alternative<ArgSlot>(arg)) {
                    mlir << slot_to_ssa.at(std::get<ArgSlot>(arg).slot);
                } else if (std::holds_alternative<ArgLit>(arg)) {
                    // For literals, we'd need to emit a constant op first
                    // For now, emit a placeholder
                    mlir << "'%'lit_" << i;
                }
            }
            
            // Emit type signature
            mlir << " : ";
            
            // For binary ops, emit both operand types
            if (step.args.size() == 2) {
                // Simplified: assume same type for both operands
                mlir << ag::jit::mlir_emit::shapeToMLIRType(step.out_meta.shape, step.out_meta.dtype) << ", "
                     << ag::jit::mlir_emit::shapeToMLIRType(step.out_meta.shape, step.out_meta.dtype);
            } else if (step.args.size() == 1) {
                // Unary op
                mlir << ag::jit::mlir_emit::shapeToMLIRType(step.out_meta.shape, step.out_meta.dtype);
            }
        }
        
        mlir << "\n";
    }
    
    // Emit return statement
    mlir << "  return " << slot_to_ssa.at(plan.out_slot) 
         << " : " << ag::jit::mlir_emit::shapeToMLIRType(plan.steps.back().out_meta.shape, plan.steps.back().out_meta.dtype) << "\n";
    
    mlir << "}\n";
    
    return mlir.str();
}

} // namespace mlir_emit

} // namespace ag::jit


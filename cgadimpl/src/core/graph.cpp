// =====================
// file: cgadimpl/src/graph.cpp
// =====================
#include "ad/core/graph.hpp"
#include <unordered_set>
#include <functional>
#include <cassert>
#include <sstream>
#include <iostream> // Added for printing


namespace ag {

// --- Node Implementation ---
// Node::Node() = default; 
Node::Node(const Tensor& v, Op op_, bool req_grad, const char* nm) 
    : op(op_), 
      value(v),
      requires_grad_flag_(req_grad),
      debug_name(nm),
      is_leaf(op_ == Op::Leaf)  // Phase 1.1: Mark leaf nodes
{
    // Phase 1.3: Capture execution context
    creation_context.stream = current_stream();
    creation_context.device = v.device();
    
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
        // grad = Tensor(Shape{}, TensorOptions().with_dtype(v.dtype()).with_device(v.device()));
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
#include <ad/runtime/runtime.hpp>
#include <ad/mlir_emitter.hpp>
#include "Compiler/API/NovaCompilerAPI.h"
#include "mlir/IR/BuiltinOps.h"  // For ModuleOp definition

namespace ag::jit {
// Use definitions from mlir_emitter.hpp

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


// Helper to convert Dtype to MLIR tensor type string
static std::string dtypeToMLIR(Dtype dt) {
    switch (dt) {
        case OwnTensor::Dtype::Float32:  return "f32";
        case OwnTensor::Dtype::Float16:  return "f16";
        case OwnTensor::Dtype::Bfloat16: return "bf16";
        case OwnTensor::Dtype::Int32:    return "i32";
        case OwnTensor::Dtype::Int64:    return "i64";
        default:              return "unknown";
    }
}

// Helper to format shape as string
static std::string shapeToMLIR(const std::vector<int64_t>& shape) {
    std::string s;
    for (int64_t dim : shape) {
        s += std::to_string(dim) + "x";
    }
    // if (!s.empty()) s.pop_back();
    return s;
}

// Helper function to map your Op enum to a specific 'nova' dialect operation name
// This mapping is crucial to match your desired output format
static std::string opToNovaOp(Op op) {
    switch (op) {
        case Op::Add:       return "nova.add";
        case Op::Mul:       return "nova.mul";
        case Op::MatMul:    return "nova.matmul"; 
        case Op::Sum:       return "nova.reduce<sum>";
        case Op::MeanAll:   return "nova.reduce<mean>";
        default:            return "nova.unknown_op";
    }
}


/// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////*************************** */
static std::string emitMLIR(const Plan& plan) {
    std::stringstream ss;
    
    // --- Function Signature ---
    ss << "func.func @main(";
    
    // Define all external inputs/params as %arg0, %arg1, ...
    size_t arg_idx_counter = 0;

    auto print_arg_meta = [&](const std::vector<TensorMetadata>& metas) {
        for (size_t i = 0; i < metas.size(); ++i) {
            const auto& meta = metas[i];
            ss << "%arg" << arg_idx_counter++ << ": tensor<" 
               << shapeToMLIR(meta.shape) << dtypeToMLIR(meta.dtype) << ">";
            if (i < metas.size() -1  || !plan.sig.param_meta.empty()) ss << "";
        }
    };

    print_arg_meta(plan.sig.in_meta);
    // print_arg_meta(plan.sig.param_meta); // Enable if params should also be func args

    ss << ") -> tensor<" 
       << shapeToMLIR(plan.steps.back().out_meta.shape) 
       << dtypeToMLIR(plan.steps.back().out_meta.dtype) << "> {\n";


    // --- Function Body (Steps) ---
    std::unordered_map<int, std::string> slot_to_var_name; // Maps plan slot IDs to MLIR variable names (%v0, %v1, %a, %b)
    std::unordered_map<int, TensorMetadata> slot_to_meta; // Maps plan slot IDs to their metadata for type lookup

    // Store metadata for slots created so far (needed for backward lookup)
    for (const auto& st : plan.steps) {
        slot_to_meta[st.out_slot] = st.out_meta;
    }

    for (size_t i = 0; i < plan.steps.size(); ++i) {
        const auto& st = plan.steps[i];
        std::string result_var = "%v" + std::to_string(i); // Using %v0, %v1 for now
        slot_to_var_name[st.out_slot] = result_var;

        ss << "  " << result_var << " = " << opToNovaOp(st.op) << " ";

        std::vector<std::string> arg_names;
        std::vector<std::string> arg_types;

        // Process arguments
        for (const auto& arg : st.args) {
            std::visit([&](auto&& a) {
                using T = std::decay_t<decltype(a)>;
                if constexpr (std::is_same_v<T, ArgInput> || std::is_same_v<T, ArgParam>) {
                    // This assumes inputs and params are mapped correctly to %arg0, %arg1, ...
                    // If you have params, the index calculation needs adjustment.
                    int arg_idx = a.idx; 
                    // Simplified lookup for demo: assumes all inputs/params indexed linearly from 0 in sig
                    const auto& meta = (std::is_same_v<T, ArgInput>) ? plan.sig.in_meta[arg_idx] : plan.sig.param_meta[arg_idx];
                    arg_names.push_back("%arg" + std::to_string(arg_idx));
                    arg_types.push_back("tensor<" + shapeToMLIR(meta.shape) + dtypeToMLIR(meta.dtype) + ">");
                } else if constexpr (std::is_same_v<T, ArgSlot>) {
                    arg_names.push_back(slot_to_var_name.at(a.slot));
                    // Look up the type from the stored slot metadata
                    const auto& meta = slot_to_meta.at(a.slot);
                    arg_types.push_back("tensor<" + shapeToMLIR(meta.shape) + dtypeToMLIR(meta.dtype) + ">");
                } else if constexpr (std::is_same_v<T, ArgLit>) {
                    arg_names.push_back("const_lit"); 
                    arg_types.push_back("tensor<f32>"); // Placeholder for literals in fallback
                }
            }, arg);
        }

        // Print arguments and types in the required format
        for (size_t j = 0; j < arg_names.size(); ++j) {
            ss << arg_names[j];
            if (j < arg_names.size() - 1) ss << ", ";
        }
        
        ss << ": ";

         for (size_t j = 0; j < arg_types.size(); ++j) {
            ss << arg_types[j];
            if (j < arg_types.size() - 1) ss << ", ";
        }
        
        ss << "\n";
    }

    // --- Return Statement ---
    std::string return_var = slot_to_var_name.at(plan.out_slot);
    const auto& return_meta = plan.steps.back().out_meta;
    auto return_shape = return_meta.shape;
    
    // Total reduction rank adjustment for fallback
    if ((plan.steps.back().op == Op::Sum || plan.steps.back().op == Op::MeanAll) && 
        return_shape.size() == 1 && return_shape[0] == 1) {
        return_shape = {};
    }

    ss << "  return " << return_var << " : tensor<"
       << shapeToMLIR(return_shape) 
       << dtypeToMLIR(return_meta.dtype) << ">\n";
       
    ss << "}\n";

    return ss.str();
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Compiled compile(const Value& output,
                 const std::vector<Value>& inputs,
                 const std::vector<Value>& params,
                 const CompileOptions&) {
    // Map externals (no changes needed here)
    std::unordered_map<Node*,int> in_ix, par_ix;
    for (size_t i = 0; i < inputs.size(); ++i) in_ix[inputs[i].node.get()] = i;
    for (size_t i = 0; i < params.size(); ++i) par_ix[params[i].node.get()] = i;
    // ...

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

    // --- Generate MLIR using the new C++ API approach ---
    std::string generated_mlir_opbuilder;
    mlir::OwningOpRef<mlir::ModuleOp> in_memory_module;
    std::shared_ptr<mlir::MLIRContext> context;
    
    try {
        MLIREmitter emitter;
        context = emitter.getContext();
        auto [module, mlirStr] = emitter.emitModule(plan);
        generated_mlir_opbuilder = mlirStr;
        in_memory_module = std::move(module);  // Store in-memory module!
        std::cout << "\n=== MLIR Generated via OpBuilder ===\n" << generated_mlir_opbuilder << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Warning: MLIR OpBuilder emission failed: " << e.what() << "\n";
        std::cerr << "Falling back to string-based emission\n";
    }

    // --- Fallback: Generate MLIR source using string-based approach ---
    std::string generated_mlir_string = emitMLIR(plan);
    if (generated_mlir_opbuilder.empty()) {
        std::cout << "\n=== MLIR Generated via String (Fallback) ===\n" << generated_mlir_string << std::endl;
    }

    Compiled c;
    c.p = std::make_shared<Compiled::Impl>();
    c.p->plan = std::move(plan);
    c.mlir_source = std::move(generated_mlir_string);
    
    // Store in-memory MLIR module (wrap in shared_ptr with custom deleter)
    if (in_memory_module) {
        // --- Run Nova Optimization Pipeline ---
        try {
            mlir::nova::NovaCompilerAPI compiler;
            mlir::nova::CompilerOptions options;
            options.runFullPipeline = true;
            
            // We use the string output from MLIREmitter to avoid context mismatch issues
            auto compileResult = compiler.compileString(generated_mlir_opbuilder, "", options);
            if (compileResult.success) {
                generated_mlir_opbuilder = compileResult.output;
                std::cout << "\n=== Optimized MLIR Generated via NovaCompilerAPI ===\n" << generated_mlir_opbuilder << std::endl;
            } else {
                std::cerr << "Warning: NovaCompilerAPI pipeline failed: " << compileResult.errorMessage << "\n";
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: NovaCompilerAPI integration failed: " << e.what() << "\n";
        }

        auto* module_ptr = new mlir::OwningOpRef<mlir::ModuleOp>(std::move(in_memory_module));
        c.mlir_module = std::shared_ptr<void>(module_ptr, [context](void* p) {
            delete static_cast<mlir::OwningOpRef<mlir::ModuleOp>*>(p);
        });
    }

    c.mlir_module_str = std::move(generated_mlir_opbuilder);  // Final (potentially optimized) string
    
    return c;
}

bool Compiled::run(const std::vector<Tensor*>& inputs,
                   const std::vector<Tensor*>& params,
                   Tensor& out) const {
    return p->run(inputs, params, out);
}

const std::string& Compiled::getMLIRSource() const {
    return mlir_source;
}

void* Compiled::getMLIRModule() const {
    if (mlir_module) {
        auto* module_ptr = static_cast<mlir::OwningOpRef<mlir::ModuleOp>*>(mlir_module.get());
        if (module_ptr && module_ptr->get()) {
            return module_ptr->get();
        }
    }
    return nullptr;
}


} // namespace ag::jit
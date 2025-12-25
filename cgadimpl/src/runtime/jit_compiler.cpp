#include "ad/runtime/jit_compiler.hpp"
#include "ad/ops/nodeops.hpp" 
#include "TensorLib.h"
#include <unordered_map>
#include <variant>
#include <iostream>
#include <cassert>

namespace ag::jit {

using ag::Op;
using ag::Node;
// using ag::Tensor;

// ===================================================================
// JIT Compiler Implementation
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
            case Op::Relu:     { return (*a[0] + OwnTensor::abs(*a[0], ag::current_stream())) * 0.5f;}
            case Op::Exp:        return OwnTensor::exp(*a[0]);
            case Op::Log:        return OwnTensor::log(*a[0]);
            case Op::Tanh:       return OwnTensor::tanh(*a[0]);
            
            case Op::MatMul:     return OwnTensor::matmul(*a[0], *a[1]);

            // Reductions need to be updated to the new API
            case Op::Sum: return OwnTensor::reduce_sum(*a[0]);
            case Op::RowSum: return OwnTensor::reduce_sum(*a[0], {1}, true);
            case Op::RowMax: return OwnTensor::reduce_max(*a[0], {1}, true);
            case Op::MeanAll: return OwnTensor::reduce_mean(*a[0]);

            case Op::Leaf: default: {
                // Shouldn't get called for Leaf
                // assert(false && "apply(): unexpected op");
                return *a[0];
            }
        }
    }

    bool run(const std::vector<Tensor*>& inputs,
             const std::vector<Tensor*>& params,
             Tensor& out) const {
        if (!plan.sig.matches(inputs, params)) return false;

        std::vector<Tensor> slots(plan.num_slots);
        
        for (const Step& st : plan.steps) {
            if (st.out_slot >= 0) {
                slots[st.out_slot] = Tensor(OwnTensor::Shape{st.out_meta.shape}, st.out_meta.dtype, st.out_meta.device, false);
            }
        }

        // Execute
        for (const Step& st : plan.steps) {
            std::vector<const Tensor*> args; args.reserve(st.args.size());
            
            Tensor tmp{OwnTensor::Shape{}, OwnTensor::TensorOptions{}}; 
            
            std::vector<Tensor> tmp_keep; tmp_keep.reserve(st.args.size());
            for (const Arg& a : st.args) {
                if (std::holds_alternative<ArgLit>(a)) {
                    tmp_keep.emplace_back(std::get<ArgLit>(a).t);
                    args.push_back(&tmp_keep.back());
                } else {
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
    for(size_t i=0;i<inputs.size();++i) in_ix[inputs[i].node.get()]=i;
    for(size_t i=0;i<params.size();++i) par_ix[params[i].node.get()]=i;

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
    
    // Debug printing omitted for brevity in refactored file, but logic remains
    
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

// =====================
// file: src/graph.cpp
// =====================
#include <unordered_set>
#include <functional>
#include <cassert>
#include "ad/graph.hpp"
#include "nn/nn.hpp" // for silu


namespace ag {


    Node::Node() = default;
    Node::Node(const Tensor& v, bool rg, Op op_, const char* nm) : op(op_), value(v), grad(Tensor::zeros_like(v)), requires_grad(rg), debug_name(nm) {}


    Value::Value() = default;

    Value::Value(std::shared_ptr<Node> n): node(std::move(n)) {}

    const Tensor& Value::val() const { 
        return node->value; 
    }

    Tensor& Value::grad() { 
        return node->grad; 
    }
    
    std::pair<int,int> Value::shape() const { 
        return node->value.shape(); 
    }


    Value constant(const Tensor& v, const char* name){ 
        return Value(std::make_shared<Node>(v,false,Op::Leaf,name)); 
    }

    Value param (const Tensor& v, const char* name){ 
        return Value(std::make_shared<Node>(v,true ,Op::Leaf,name));
    }

    std::vector<Node*> topo_from(Node* root){
        std::vector<Node*> order; order.reserve(256);
        std::unordered_set<Node*> vis; vis.reserve(256);
        std::function<void(Node*)> dfs = [&](Node* n){ if(!n || vis.count(n)) return; vis.insert(n); for(auto& p : n->inputs) dfs(p.get()); order.push_back(n); };
        dfs(root);
        return order; // parents before child
    }


} // namespace ag

// ===================== ag::jit implementation =====================
#include <unordered_map>
#include <variant>

namespace ag::jit {

struct Signature {
    std::vector<std::pair<int,int>> in_shapes;
    std::vector<std::pair<int,int>> param_shapes;
    bool matches(const std::vector<Tensor*>& inputs,
                 const std::vector<Tensor*>& params) const {
        if (inputs.size() != in_shapes.size() || params.size() != param_shapes.size())
            return false;
        for (size_t i=0;i<inputs.size();++i){
            auto s = inputs[i]->shape();
            if (s != in_shapes[i]) return false;
        }
        for (size_t i=0;i<params.size();++i){
            auto s = params[i]->shape();
            if (s != param_shapes[i]) return false;
        }
        return true;
    }
};

// Arg sources for a Step
struct ArgInput  { int idx; };   // external input[i]
struct ArgParam  { int idx; };   // external param[i]
struct ArgSlot   { int slot; };  // prior computed slot
struct ArgLit    { Tensor t; };  // embedded literal

using Arg = std::variant<ArgInput,ArgParam,ArgSlot,ArgLit>;

struct Step {
    Op op;
    std::vector<Arg> args;
    int out_slot{};                 // where to write result
    std::pair<int,int> out_shape{}; // rows,cols
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

    static Tensor broadcast_to(const Tensor& X, int R, int C) {
        if (X.rows()==R && X.cols()==C) return X;
        Tensor Y(R,C);
        if (X.rows()==1 && X.cols()==1) {
            for(int i=0;i<R;++i) for(int j=0;j<C;++j) Y(i,j)=X(0,0);
            return Y;
        }
        if (X.rows()==R && X.cols()==1) { // column bias
            for(int i=0;i<R;++i){ float v=X(i,0); for(int j=0;j<C;++j) Y(i,j)=v; }
            return Y;
        }
        if (X.rows()==1 && X.cols()==C) { // row vector
            for(int i=0;i<R;++i) for(int j=0;j<C;++j) Y(i,j)=X(0,j);
            return Y;
        }
        // fallback: require exact
        assert(false && "broadcast_to: incompatible shapes");
        return X;
    }

    static Tensor mul_scalar(const Tensor& X, float s){
        Tensor Y(X.rows(), X.cols());
        for(int i=0;i<X.rows();++i) for(int j=0;j<X.cols();++j) Y(i,j) = X(i,j)*s;
        return Y;
    }

    static Tensor apply(Op op, const std::vector<const Tensor*>& a) {
        // a.size() equals op_arity(op), except literals we materialized as tensors
        switch(op){
            case Op::Add: {
                int R=a[0]->rows(), C=a[0]->cols();
                Tensor A = broadcast_to(*a[0], R, C);
                Tensor B = broadcast_to(*a[1], R, C);
                return A + B;
            }
            case Op::Sub: {
                int R=a[0]->rows(), C=a[0]->cols();
                Tensor A = broadcast_to(*a[0], R, C);
                Tensor B = broadcast_to(*a[1], R, C);
                return A - B;
            }
            case Op::Mul: {
                int R=a[0]->rows(), C=a[0]->cols();
                Tensor A = broadcast_to(*a[0], R, C);
                Tensor B = broadcast_to(*a[1], R, C);
                return A * B;
            }
            case Op::Transpose:  return Tensor::transpose(*a[0]);
            case Op::Relu:       return nn::relu(*a[0]);
            case Op::Exp:        return Tensor::exp (*a[0]);
            case Op::Log:        return Tensor::log (*a[0]);
            case Op::Tanh:       return nn::tanh(*a[0]);
            case Op::Sigmoid:    return nn::sigmoid(*a[0]);
            case Op::Softplus:   return nn::softplus(*a[0]);
            case Op::SiLU:       return nn::silu(*a[0]);
            case Op::GELU:       return nn::gelu(*a[0]);
            case Op::LeakyRelu: {
                // args: X, alpha (alpha can be literal 1x1 or input/param 1x1)
                int R=a[0]->rows(), C=a[0]->cols();
                Tensor Y(R,C);
                float alpha = (*a[1])(0,0);
                for(int i=0;i<R;++i) for(int j=0;j<C;++j){
                    float x = (*a[0])(i,j);
                    Y(i,j) = (x>0.f) ? x : alpha*x;
                }
                return Y;
            }
            case Op::MatMul:     return Tensor::matmul(*a[0], *a[1]);
            case Op::Sum:        return Tensor::sum_all(*a[0]);
            case Op::RowSum:     return Tensor::row_sum(*a[0]);
            case Op::RowMax:     return Tensor::row_max(*a[0]);
            case Op::MeanAll: {
                Tensor s = Tensor::sum_all(*a[0]);
                float inv = 1.f / float(a[0]->rows()*a[0]->cols());
                return mul_scalar(s, inv);
            }
            case Op::SoftmaxRow: return nn::softmax_row(*a[0]);
            case Op::LogSumExpRow: return nn::logsumexp_row(*a[0]);
            case Op::CeWithLogits: {
                // CE = -mean( sum( Y * (Z - lse(Z)), axis=1 ) )
                const Tensor& Z = *a[0];
                const Tensor& Y = *a[1];
                Tensor lse = nn::logsumexp_row(Z);           // [B,1]
                // broadcast lse to [B,C]
                Tensor L = broadcast_to(lse, Z.rows(), Z.cols());
                Tensor term = Y * (Z - L);                        // [B,C]
                Tensor rs = Tensor::row_sum(term);                // [B,1]
                Tensor s = Tensor::sum_all(rs);                   // [1,1]
                float invB = -1.f / float(Z.rows());
                return mul_scalar(s, invB);
            }
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
        // Preallocate slot tensors with correct shapes
        for (const Step& st : plan.steps) {
            if (st.out_slot >= 0) {
                slots[st.out_slot] = Tensor(st.out_shape.first, st.out_shape.second);
            }
        }

        // Execute
        for (const Step& st : plan.steps) {
            std::vector<const Tensor*> args; args.reserve(st.args.size());
            Tensor tmp; // for literal materialization
            std::vector<Tensor> tmp_keep; tmp_keep.reserve(st.args.size()); // keep lifetimes
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
    in_ix.reserve(inputs.size()); par_ix.reserve(params.size());
    for (size_t i=0;i<inputs.size(); ++i) in_ix[ inputs[i].node.get() ] = int(i);
    for (size_t i=0;i<params.size(); ++i) par_ix[ params[i].node.get() ] = int(i);

    // Build plan
    Plan plan;
    plan.sig.in_shapes.reserve(inputs.size());
    for (auto& v: inputs)  plan.sig.in_shapes.push_back(v.val().shape());
    plan.sig.param_shapes.reserve(params.size());
    for (auto& v: params)  plan.sig.param_shapes.push_back(v.val().shape());

    auto order = topo_from(output.node.get());
    std::unordered_map<Node*,int> slot_of;
    slot_of.reserve(order.size());

    for (Node* n : order) {
        if (n->op == Op::Leaf) {
            // Leaves are sources; nothing to emit. They get materialized as ArgInput/ArgParam or ArgLit where used.
            continue;
        }
        Step st;
        st.op = n->op;
        st.out_shape = n->value.shape();
        st.out_slot = plan.num_slots++;
        slot_of[n] = st.out_slot;

        // Gather args
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

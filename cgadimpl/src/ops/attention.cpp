#include "ad/ops/nodeops.hpp"
#include <cuda_runtime.h>
#include "tensor.hpp" 
#include <unordered_map>
#include <cmath> 
#include <type_traits> 

namespace ag {
namespace detail {
std::shared_ptr<Node> attention_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c, const std::shared_ptr<Node>& d){
    Tensor q = matmul(a->value, b->value);
    Tensor k = matmul(a->value, c->value);
    Tensor v = matmul(a->value, d->value);
    float scale = 1.f / sqrtf(static_cast<float>(k.shape().dims.back()));
    Tensor g = matmul(q, k.t()) * scale;
    Tensor max_val = reduce_max(g, {-1}, true);
    Tensor exp_g = exp(g - max_val, ag::current_stream());
    Tensor sum_exp_g = reduce_sum(exp_g, {-1}, true);
    Tensor s = exp_g / sum_exp_g;
    Tensor y = matmul(s, v);
    auto n = std::make_shared<Node>(y, Op::Attention, (a->requires_grad() || b->requires_grad() || c->requires_grad() || d->requires_grad()), "attention");
    n->inputs = {a, b, c, d};
    n->tape.push_back(std::make_shared<Tensor>(q));
    n->tape.push_back(std::make_shared<Tensor>(k));
    n->tape.push_back(std::make_shared<Tensor>(v));
    n->tape.push_back(std::make_shared<Tensor>(s));
    if(a) a->child_grad_count++;
    if(b) b->child_grad_count++;
    if(c) c->child_grad_count++;
    if(d) d->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}
std::shared_ptr<Node> swiglu_nodeops(const std::shared_ptr<Node>& x, const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c, const std::shared_ptr<Node>& d){ 
    Tensor y = OwnTensor::matmul(x->value, a->value.t()) + b->value; 
    Tensor q = y * (1.0f / (1.0f + OwnTensor::exp(y * -1.0f, ag::current_stream())));
    Tensor w = q * (OwnTensor::matmul(x->value, c->value.t()) + d->value);
    auto n = std::make_shared<Node>(w, Op::SWIGLU, (x->requires_grad() || a->requires_grad() || b->requires_grad() || c->requires_grad() || d-> requires_grad()) , "swiglu"); 
    n->inputs={x, a, b, c, d};
    if (x) x->child_grad_count++;
    if (a) a->child_grad_count++;
    if (b) b->child_grad_count++;
    if (c) c->child_grad_count++;
    if (d) d->child_grad_count++;
    ag::debug::on_node_created(n); 
    return n;
}
} // namespace detail
} // namespace ag
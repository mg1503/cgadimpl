#include "ad/ops/nodeops.hpp"
#include <cuda_runtime.h>
#include "tensor.hpp" 
#include <unordered_map>
#include <cmath> 
#include <type_traits> 

namespace ag {
namespace detail {

std::shared_ptr<Node> add_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){
    Tensor Y = a->value + b->value; 
    auto n = std::make_shared<Node>(Y, Op::Add, (a->requires_grad() || b->requires_grad()), "+");
    n->inputs = {a, b};
    if(a) a->child_grad_count++;
    if(b) b->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}
  
std::shared_ptr<Node> sub_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){
    Tensor Y = a->value - b->value;
    auto n = std::make_shared<Node>(Y, Op::Sub, (a->requires_grad() || b->requires_grad()), "-");
    n->inputs = {a, b};
    if(a) a->child_grad_count++;
    if(b) b->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}

std::shared_ptr<Node> mul_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){ 
    Tensor y = a->value * b->value; 
    auto n = std::make_shared<Node>(y, Op::Mul, (a->requires_grad() || b->requires_grad()), "*"); 
    n->inputs = {a, b}; 
    if(a) a->child_grad_count++;
    if(b) b->child_grad_count++;
    ag::debug::on_node_created(n); 
    return n; 
}

std::shared_ptr<Node> div_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){
    Tensor C = a->value / b->value;
    auto n = std::make_shared<Node>(C, Op::Div, (a->requires_grad() || b->requires_grad()), "/");
    n->inputs = { a, b };
    if(a) a->child_grad_count++;
    if(b) b->child_grad_count++;
    ag::debug::on_node_created(n);  
    return n;
}

std::shared_ptr<Node> exp_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = OwnTensor::exp(x->value, ag::current_stream());
    auto n = std::make_shared<Node>(y, Op::Exp, x->requires_grad(), "exp");
    n->inputs = {x};
    if(x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}

std::shared_ptr<Node> log_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = OwnTensor::log(x->value, ag::current_stream());
    auto n = std::make_shared<Node>(y, Op::Log, x->requires_grad(), "log");
    n->inputs = {x};
    if(x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}
std::shared_ptr<Node> sqrt_nodeops(const std::shared_ptr<Node>& x) {
    Tensor y = OwnTensor::sqrt(x->value, ag::current_stream());
    auto n = std::make_shared<Node>(y, Op::Sqrt, x->requires_grad(), "sqrt");
    n->inputs = {x};
    if (x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}
std::shared_ptr<Node> reci_nodeops(const std::shared_ptr<Node>& a) {
    Tensor y = 1.0f / a->value;
    auto n = std::make_shared<Node>(y, Op::Reciprocal, a->requires_grad(),"reciprocal");
    n->inputs = {a};
    if(a) a->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}
std::shared_ptr<Node> sign_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = OwnTensor::sign(x->value, ag::current_stream());
    auto n = std::make_shared<Node>(y, Op::Sign, x->requires_grad(), "sign");
    n->inputs={x};
    if (x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}
std::shared_ptr<Node> abs_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = OwnTensor::abs(x->value, ag::current_stream());
    auto n = std::make_shared<Node>(y, Op::Abs, x->requires_grad(), "abs");
    n->inputs={x};
    if(x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}
std::shared_ptr<Node> pow_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){
    // OwnTensor::pow only supports scalar exponents. 
    // For tensor exponents, we use: a^b = exp(b * log(a))
    Tensor Y = OwnTensor::exp(b->value * OwnTensor::log(a->value, ag::current_stream()), ag::current_stream());
    auto n = std::make_shared<Node>(Y, Op::Pow, (a->requires_grad() || b->requires_grad()), "pow");
    n->inputs = { a, b };
    if(a) a->child_grad_count++;
    if(b) b->child_grad_count++;
    ag::debug::on_node_created(n);  
    return n;
}

} // namespace detail
} // namespace ag
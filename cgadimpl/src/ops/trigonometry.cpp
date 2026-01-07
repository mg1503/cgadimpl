#include "ad/ops/nodeops.hpp"
#include <cuda_runtime.h>
#include "tensor.hpp" 
#include <unordered_map>
#include <cmath> 
#include <type_traits> 

namespace ag {
namespace detail {
std::shared_ptr<Node> sin_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = trig::sin(x->value);
    auto n=std::make_shared<Node>(y, Op::Sin, x->requires_grad(), "sin");
    n->inputs={x};
    if(x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}
std::shared_ptr<Node> cos_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = trig::cos(x->value);
    auto n=std::make_shared<Node>(y, Op::Cos, x->requires_grad(), "cos");
    n->inputs={x};
    if (x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}
std::shared_ptr<Node> tan_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = trig::tan(x->value);
    auto n=std::make_shared<Node>(y, Op::Tan, x->requires_grad(), "tan");
    n->inputs={x};
    if (x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}


std::shared_ptr<Node> sinh_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = trig::sinh(x->value);
    auto n=std::make_shared<Node>(y, Op::Sinh, x->requires_grad(), "sinh");
    n->inputs={x};
    if (x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}
std::shared_ptr<Node> cosh_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = trig::cosh(x->value);
    auto n=std::make_shared<Node>(y, Op::Cosh, x->requires_grad(), "cosh");
    n->inputs={x};
    if (x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}

std::shared_ptr<Node> asin_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = trig::asin(x->value);
    auto n=std::make_shared<Node>(y, Op::Asin, x->requires_grad(), "asin");
    n->inputs={x};
    if(x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}
std::shared_ptr<Node> acos_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = trig::acos(x->value);
    auto n=std::make_shared<Node>(y, Op::Acos, x->requires_grad(), "acos");
    n->inputs={x};
    if(x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}
std::shared_ptr<Node> atan_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = trig::atan(x->value);
    auto n=std::make_shared<Node>(y, Op::Atan, x->requires_grad(), "atan");
    n->inputs={x};
    if(x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}

std::shared_ptr<Node> asinh_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = trig::asinh(x->value);
    auto n=std::make_shared<Node>(y, Op::ASinh, x->requires_grad(), "asinh");
    n->inputs={x};
    if(x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}
std::shared_ptr<Node> acosh_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = trig::acosh(x->value);
    auto n=std::make_shared<Node>(y, Op::ACosh, x->requires_grad(), "acosh");
    n->inputs={x};
    if(x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}
std::shared_ptr<Node> atanh_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = trig::atanh(x->value);
    auto n=std::make_shared<Node>(y, Op::ATanh, x->requires_grad(), "atanh");
    n->inputs={x};
    if(x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}
} // namespace detail
} // namespace ag
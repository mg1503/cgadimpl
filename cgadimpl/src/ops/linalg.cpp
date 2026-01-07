#include "ad/ops/nodeops.hpp"
#include <cuda_runtime.h>
#include "tensor.hpp" 
#include <unordered_map>
#include <cmath> 
#include <type_traits> 
#include "mlp/layers.h"

namespace ag {
namespace detail {


std::shared_ptr<Node> matmul_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b) {
    Tensor C = matmul(a->value, b->value);
    auto n = std::make_shared<Node>(C, Op::MatMul, (a->requires_grad() || b->requires_grad()), "matmul");
    n->inputs = {a, b};
    if(a) a->child_grad_count++;
    if(b) b->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}
std::shared_ptr<Node> linear_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c) {
    const Tensor& input_X = a->value;
    const Tensor& weight_W = b->value; 
    const Tensor& bias_b = c->value;
    // Tensor y = matmul(input_X, weight_W.t()) + bias_b;
    Tensor y = OwnTensor::mlp_forward::linear(input_X, weight_W, bias_b);
    auto n = std::make_shared<Node>(y, Op::Linear, (a->requires_grad() || b->requires_grad() || c->requires_grad()), "linear");
    n->inputs = {a, b, c};
    if (a) a->child_grad_count++;
    if (b) b->child_grad_count++;
    if (c) c->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}

std::shared_ptr<Node> dropout_nodeops(const std::shared_ptr<Node>& a, // Input X (0, 1)
                                     const std::shared_ptr<Node>& b) // float p
{
    const Tensor& input = a->value;
    const Tensor& p_tensor = b->value;

    // Extract the float value
    float p_val = *p_tensor.template data<float>();

    Tensor y = OwnTensor::mlp_forward::dropout(input, p_val);

    auto n = std::make_shared<Node>(y, Op::Dropout, (a->requires_grad() || b->requires_grad()), "dropout");
    n->inputs = {a, b};
    n->tape.push_back(std::make_shared<Tensor>(y));

    // NEW CODE LINES--> DEPENDENCY COUNTER
    if (a) a->child_grad_count++;
    if (b) b->child_grad_count++;

    ag::debug::on_node_created(n);
    return n;
}

std::shared_ptr<Node> flatten_nodeops(const std::shared_ptr<Node>& a) // Input a
{
    const Tensor& input_X = a->value;

    Tensor y = OwnTensor::mlp_forward::flatten(input_X);

    auto n = std::make_shared<Node>(y, Op::Flatten, (a->requires_grad()), "flatten");
    n->inputs = {a};

    if (a) a->child_grad_count++;

    ag::debug::on_node_created(n);
    return n;
}

std::shared_ptr<Node> fmab_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c){
    Tensor y = matmul(a->value, b->value) + c->value;
    auto n = std::make_shared<Node>(y, Op::FMA, (a->requires_grad() || b->requires_grad() || c->requires_grad()), "fmab");
    n->inputs = {a, b, c};
    if(a) a->child_grad_count++;
    if(b) b->child_grad_count++;
    if(c) c->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}

std::shared_ptr<Node> transpose_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = x->value.t();
    auto n = std::make_shared<Node>(y, Op::Transpose, x->requires_grad(), "transpose");
    n->inputs = {x};
    if(x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}

} // namespace detail
} // namespace ag
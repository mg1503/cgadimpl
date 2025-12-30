#include "ad/ops/nodeops.hpp"
#include <cuda_runtime.h>
#include "tensor.hpp" 
#include <unordered_map>
#include <cmath> 
#include <type_traits> 
namespace ag {
namespace detail {
    std::shared_ptr<Node> sum_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = OwnTensor::reduce_sum(x->value, {}, false);
    auto n = std::make_shared<Node>(y, Op::Sum, x->requires_grad(), "sum");
    n->inputs = {x};
    if(x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}
std::shared_ptr<Node> mean_all_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = OwnTensor::reduce_mean(x->value);
    auto n = std::make_shared<Node>(y, Op::MeanAll, x->requires_grad(), "meanall");
    n->inputs={x};
    if(x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}
std::shared_ptr<Node> rowsum_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = OwnTensor::reduce_sum(x->value, {1}, true);
    auto n = std::make_shared<Node>(y, Op::RowSum, x->requires_grad(), "rowsum");
    n->inputs = {x};
    if(x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}
std::shared_ptr<Node> rowmax_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = OwnTensor::reduce_max(x->value, {1}, true);
    auto n = std::make_shared<Node>(y, Op::RowMax, x->requires_grad(), "rowmax");
    n->inputs={x};
    if(x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}

std::shared_ptr<Node> softmax_row_nodeops(const std::shared_ptr<Node>& z){ 
    Tensor max_val = OwnTensor::reduce_max(z->value, {-1}, true);
    Tensor z_shifted = z->value - max_val;
    Tensor exp_z = OwnTensor::exp(z_shifted, ag::current_stream());
    Tensor sum_exp_z = OwnTensor::reduce_sum(exp_z, {-1}, true);
    Tensor y = exp_z / sum_exp_z;
    auto n = std::make_shared<Node>(y, Op::SoftmaxRow, z->requires_grad(), "softmax_row"); 
    n->inputs = {z}; 
    if (z) z->child_grad_count++;
    ag::debug::on_node_created(n);  
    return n;
}
std::shared_ptr<Node> logsumexp_row_nodeops(const std::shared_ptr<Node>& z){ 
    Tensor max_val = OwnTensor::reduce_max(z->value, {-1}, true);
    Tensor z_shifted = z->value - max_val;
    Tensor exp_z = OwnTensor::exp(z_shifted, ag::current_stream());
    Tensor sum_exp_z = OwnTensor::reduce_sum(exp_z, {-1}, true);
    Tensor log_sum = OwnTensor::log(sum_exp_z, ag::current_stream());
    Tensor y = log_sum + max_val;
    auto n = std::make_shared<Node>(y, Op::LogSumExpRow, z->requires_grad(), "logsumexp_row"); 
    n->inputs = {z}; 
    if (z) z->child_grad_count++;
    ag::debug::on_node_created(n);  
    return n;
}
} // namespace detail
} // namespace ag
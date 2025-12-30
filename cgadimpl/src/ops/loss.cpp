#include "ad/ops/nodeops.hpp"
#include <cuda_runtime.h>
#include "tensor.hpp" 
#include <unordered_map>
#include <cmath> 
#include <type_traits> 

namespace ag {
namespace detail {
std::shared_ptr<Node> mse_loss_nodeops(const std::shared_ptr<Node>& pred, const std::shared_ptr<Node>& target) {
    Tensor diff = pred->value - target->value;
    Tensor sq   = diff * diff;
    Tensor loss = OwnTensor::reduce_mean(sq); 
    auto n = std::make_shared<Node>(loss, Op::MSELoss, (pred->requires_grad()), "mseloss");
    n->inputs = {pred, target};
    if (pred) pred->child_grad_count++;
    if (target) target->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}
std::shared_ptr<Node> mae_loss_nodeops(const std::shared_ptr<Node>& pred, const std::shared_ptr<Node>& target) {
    Tensor diff = pred->value - target->value;
    Tensor abs_diff = OwnTensor::abs(diff, ag::current_stream());
    Tensor loss = OwnTensor::reduce_mean(abs_diff);
    auto n = std::make_shared<Node>(loss, Op::MAELoss, (pred->requires_grad() || target->requires_grad()), "maeloss");
    n->inputs = {pred, target};
    if (pred) pred->child_grad_count++;
    if (target) target->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}
std::shared_ptr<Node> cross_entropy_with_logits_nodeops(const std::shared_ptr<Node>& logits, const std::shared_ptr<Node>& onehot){
    const Tensor& Z = logits->value;
    const Tensor& Y = onehot->value;
    Tensor max_val = OwnTensor::reduce_max(Z, {-1}, true);
    Tensor z_shifted = Z - max_val;
    Tensor log_sum_exp = OwnTensor::log(OwnTensor::reduce_sum(OwnTensor::exp(z_shifted, ag::current_stream()), {-1}, true), ag::current_stream());
    Tensor log_sm = z_shifted - log_sum_exp;
    Tensor prod = Y * log_sm;
    Tensor sum_prod = OwnTensor::reduce_sum(prod, {-1}); 
    Tensor loss = OwnTensor::reduce_mean(sum_prod * -1.0f); 
    auto n = std::make_shared<Node>(loss, Op::CeWithLogits, (logits->requires_grad() || onehot->requires_grad()), "ce_with_logits");
    n->inputs = {logits, onehot};
    if (logits) logits->child_grad_count++;
    if (onehot) onehot->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}

std::shared_ptr<Node> kldivergence_nodeops(const std::shared_ptr<Node>& logits, const std::shared_ptr<Node>& onehot){
    const Tensor& Z = logits->value;
    const Tensor& Y = onehot->value;
    Tensor log_Y = OwnTensor::log(Y + 1e-9f, ag::current_stream());
    Tensor max_val = OwnTensor::reduce_max(Z, {-1}, true);
    Tensor z_shifted = Z - max_val;
    Tensor log_sum_exp = OwnTensor::log(OwnTensor::reduce_sum(OwnTensor::exp(z_shifted, ag::current_stream()), {-1}, true), ag::current_stream());
    Tensor log_sm_Z = z_shifted - log_sum_exp;
    Tensor kl_div_elementwise = Y * (log_Y - log_sm_Z);
    Tensor sum_kl = OwnTensor::reduce_sum(kl_div_elementwise, {-1});
    Tensor loss = OwnTensor::reduce_mean(sum_kl);
    auto n = std::make_shared<Node>(loss, Op::KLDivergence, (logits->requires_grad() || onehot->requires_grad()), "kldivergence");
    n->inputs = {logits, onehot};
    if (logits) logits->child_grad_count++;
    if (onehot) onehot->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}

} // namespace detail
} // namespace ag
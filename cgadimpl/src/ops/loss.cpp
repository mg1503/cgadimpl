#include "ad/ops/nodeops.hpp"
#include <cuda_runtime.h>
#include "tensor.hpp" 
#include <unordered_map>
#include <cmath> 
#include <type_traits> 
#include "mlp/loss.h"

namespace ag {
namespace detail {
std::shared_ptr<Node> mse_loss_nodeops(const std::shared_ptr<Node>& pred, const std::shared_ptr<Node>& target) {
    // Tensor diff = pred->value - target->value;
    // Tensor sq   = diff * diff;
    // Tensor loss = OwnTensor::reduce_mean(sq); 
    Tensor loss = OwnTensor::mlp_forward::mse_loss(pred->value, target->value);
    auto n = std::make_shared<Node>(loss, Op::MSELoss, (pred->requires_grad()), "mseloss");
    n->inputs = {pred, target};
    if (pred) pred->child_grad_count++;
    if (target) target->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}
std::shared_ptr<Node> mae_loss_nodeops(const std::shared_ptr<Node>& pred, const std::shared_ptr<Node>& target) {
    // Tensor diff = pred->value - target->value;
    // Tensor abs_diff = OwnTensor::abs(diff, ag::current_stream());
    // Tensor loss = OwnTensor::reduce_mean(abs_diff);
    Tensor loss = OwnTensor::mlp_forward::mae_loss(pred->value, target->value);
    auto n = std::make_shared<Node>(loss, Op::MAELoss, (pred->requires_grad() || target->requires_grad()), "maeloss");
    n->inputs = {pred, target};
    if (pred) pred->child_grad_count++;
    if (target) target->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}

std::shared_ptr<Node> binary_cross_entropy_nodeops(const std::shared_ptr<Node>& pred, const std::shared_ptr<Node>& target) {

    // Tensor diff = pred->value - target->value;
    // Tensor sq   = diff * diff;
    // // --- THIS IS THE BUG ---
    // // It should be reduce_mean, not reduce_sum. reduce_mean correctly
    // // computes the VJP for the mean operation. sum has a different VJP.
    // Tensor loss = OwnTensor::reduce_mean(sq); 
    Tensor loss = OwnTensor::mlp_forward::binary_cross_entropy(pred->value, target->value);
    // --- END BUG ---

    auto n = std::make_shared<Node>(loss, Op::BinaryCrossEntropy, (pred->requires_grad() || target->requires_grad()), "binary_cross_entropy");
    n->inputs = {pred, target};

    if (pred) pred->child_grad_count++;
    if (target) target->child_grad_count++;

    ag::debug::on_node_created(n);
    return n;
}

std::shared_ptr<Node> categorical_cross_entropy_nodeops(const std::shared_ptr<Node>& pred, const std::shared_ptr<Node>& target) {

    Tensor loss = OwnTensor::mlp_forward::categorical_cross_entropy(pred->value, target->value);

    auto n = std::make_shared<Node>(loss, Op::CategoricalCrossEntropy, (pred->requires_grad() || target->requires_grad()), "categorical_cross_entropy");
    n->inputs = {pred, target};

    if (pred) pred->child_grad_count++;
    if (target) target->child_grad_count++;

    ag::debug::on_node_created(n);
    return n;
}

std::shared_ptr<Node> cross_entropy_with_logits_nodeops(const std::shared_ptr<Node>& logits, const std::shared_ptr<Node>& onehot){
    // Implementation matching user's specified formula:
    // logit_maxes = logits.max(-1, keepdim=True).values
    // norm_logits = logits - logit_maxes  # subtract max for numerical stability
    // counts = norm_logits.exp()
    // counts_sum = counts.sum(-1, keepdims=True)
    // counts_sum_inv = counts_sum**-1
    // probs = counts * counts_sum_inv
    // logprobs = probs.log()
    // loss = -logprobs[range(n), Yb].mean()  (using one-hot multiplication)
    
    const Tensor& Z = logits->value;
    const Tensor& Y = onehot->value;
    
    // logit_maxes = logits.max(-1, keepdim=True)
    Tensor logit_maxes = OwnTensor::reduce_max(Z, {-1}, true);
    
    // norm_logits = logits - logit_maxes (subtract max for numerical stability)
    Tensor norm_logits = Z - logit_maxes;
    
    // counts = norm_logits.exp()
    Tensor counts = OwnTensor::exp(norm_logits, ag::current_stream());
    
    // counts_sum = counts.sum(-1, keepdims=True)
    Tensor counts_sum = OwnTensor::reduce_sum(counts, {-1}, true);
    
    // counts_sum_inv = counts_sum**-1 (using reciprocal for exact match with PyTorch)
    Tensor counts_sum_inv = 1.0f / counts_sum;
    
    // probs = counts * counts_sum_inv
    Tensor probs = counts * counts_sum_inv;
    
    // logprobs = probs.log()
    Tensor logprobs = OwnTensor::log(probs, ag::current_stream());
    
    // loss = -logprobs[range(n), Yb].mean() (using one-hot Y to select correct indices)
    Tensor selected_logprobs = Y * logprobs;
    Tensor sum_selected = OwnTensor::reduce_sum(selected_logprobs, {-1});
    Tensor loss = OwnTensor::reduce_mean(sum_selected) * -1.0f;
    
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
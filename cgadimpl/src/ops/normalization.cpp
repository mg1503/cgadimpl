#include "ad/ops/nodeops.hpp"
#include <cuda_runtime.h>
#include "tensor.hpp" 
#include <unordered_map>
#include <cmath> 
#include <type_traits> 

namespace ag {
namespace detail {
std::shared_ptr<Node> laynor_nodeops(const std::shared_ptr<Node>& x){
    Tensor mean = OwnTensor::reduce_mean(x->value, {-1}, true);
    Tensor x_minus_mean = x->value - mean;
    Tensor variance = OwnTensor::reduce_mean(x_minus_mean * x_minus_mean, {-1}, true);
    Tensor y = x_minus_mean / OwnTensor::sqrt(variance + 1e-5f, ag::current_stream());
    auto n = std::make_shared<Node>(y, Op::LayerNorm, x->requires_grad(), "layernor");
    n->tape.push_back(std::make_shared<Tensor>(variance));
    n->tape.push_back(std::make_shared<Tensor>(mean));
    n->inputs = {x};
    if(x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}
std::shared_ptr<Node> rms_nodeops(const std::shared_ptr<Node>& x){
    Tensor x_squared = x->value * x->value;
    Tensor variance = OwnTensor::reduce_mean(x_squared, {-1}, true);
    Tensor rsqrt_var = 1.0f / OwnTensor::sqrt(variance + 1e-5f, ag::current_stream());
    Tensor y = x->value * rsqrt_var;
    auto n = std::make_shared<Node>(y, Op::RMSNorm, x->requires_grad(), "rmsnorm");
    n->tape.push_back(std::make_shared<Tensor>(rsqrt_var));
    n->tape.push_back(std::make_shared<Tensor>(y));         
    n->inputs = {x};
    if(x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}
std::shared_ptr<Node> realrms_nodeops(const std::shared_ptr<Node>& x, float& g_val){ 
    const float inv_cols = 1.0f / static_cast<float>(x->value.shape().dims.back());
    Tensor variance = OwnTensor::reduce_sum(x->value * x->value, {-1}, true) * inv_cols;
    Tensor rsqrt_var = 1.0f / OwnTensor::sqrt(variance + 1e-5f, ag::current_stream());
    Tensor y_normalized = x->value * rsqrt_var;
    static std::unordered_map<float, std::shared_ptr<Node>> scalar_cache;
    std::shared_ptr<Node> G;
    auto it = scalar_cache.find(g_val);
    if (it != scalar_cache.end()) {
        G = it->second;
    } else {
        Tensor g_tensor = Tensor::full(Shape{{1, 1}}, TensorOptions().with_req_grad(true), g_val); 
        G = std::make_shared<Node>(g_tensor, Op::Leaf, "rms_gain");
        scalar_cache[g_val] = G;
    }
    Tensor y_scaled = y_normalized * G->value;
    auto n = std::make_shared<Node>(y_scaled, Op::RealRMSNorm, x->requires_grad(), "realrmsnorm");
    n->tape.push_back(std::make_shared<Tensor>(rsqrt_var));
    n->tape.push_back(std::make_shared<Tensor>(y_normalized));
    n->inputs = {x, G};
    if(x) x->child_grad_count++;
    if(G) G->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}
std::shared_ptr<Node> dyntanh_nodeops(const std::shared_ptr<Node>& x, float& a_val, float& b_val, float& g_val){
    std::shared_ptr<Node> A = std::make_shared<Node>(Tensor::full(Shape{{1}}, TensorOptions().with_req_grad(true), a_val), Op::Leaf, "dyn_a");
    std::shared_ptr<Node> B = std::make_shared<Node>(Tensor::full(Shape{{1}}, TensorOptions().with_req_grad(true), b_val), Op::Leaf, "dyn_b");
    std::shared_ptr<Node> G = std::make_shared<Node>(Tensor::full(Shape{{1}}, TensorOptions().with_req_grad(true), g_val), Op::Leaf, "dyn_g");
    Tensor h = x->value * A->value;
    Tensor y = OwnTensor::trig::tanh(h) * G->value + B->value;
    auto n = std::make_shared<Node>(y, Op::Dyntanh, x->requires_grad(), "dyntanh");
    n->inputs={x, A, B, G};
    n->tape.push_back(std::make_shared<Tensor>(h));
    if(x) x->child_grad_count++;
    if(A) A->child_grad_count++;
    if(B) B->child_grad_count++;
    if(G) G->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}
} // namespace detail
} // namespace ag
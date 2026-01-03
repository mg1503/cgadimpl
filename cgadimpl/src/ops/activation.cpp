// =====================
// file: cgadimpl/src/ops/activation.cpp
// file: cgadimpl/src/ops/activation.cpp
// =====================
#include "ad/ops/nodeops.hpp"
#include <cuda_runtime.h>
#include "tensor.hpp" 
#include <unordered_map>
#include <cmath> 
#include <type_traits> 

namespace ag {
namespace detail {





std::shared_ptr<Node> relu_nodeops(const std::shared_ptr<Node>& x){
    const Tensor& X = x->value;
    Tensor Y = (X + OwnTensor::abs(X, ag::current_stream())) * 0.5f;
    auto n = std::make_shared<Node>(Y, Op::Relu, x->requires_grad(), "relu");
    n->inputs = {x};
    if(x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}

std::shared_ptr<Node> sigmoid_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = 1.0f / (1.0f + OwnTensor::exp(x->value * -1.0f, ag::current_stream()));
    auto n = std::make_shared<Node>(y, Op::Sigmoid, x->requires_grad(), "sigmoid"); 
    n->inputs={x}; 
    if(x) x->child_grad_count++;
    ag::debug::on_node_created(n);  
    ag::debug::on_node_created(n);  
    return n;
}
std::shared_ptr<Node> tanh_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = OwnTensor::tanh(x->value);
    auto n = std::make_shared<Node>(y, Op::Tanh, x->requires_grad(), "tanh");
    n->inputs = {x};
    if(x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}


std::shared_ptr<Node> softplus_nodeops(const std::shared_ptr<Node>& x){
    // All ops automatically use the stream from the context.
    // Tensor y = OwnTensor::log(1.0f + OwnTensor::exp(x->value));
    const float threshold = 20.0f;
    
    const Tensor& x_val = x->value;
    Tensor y = OwnTensor::Tensor::zeros(x_val.shape(), ag::options(x_val));
    
    // Dispatch by dtype to handle the computation
    dispatch_by_dtype(x_val.dtype(), [&](auto dummy){
        using T = decltype(dummy);
        if constexpr (std::is_same_v<T, OwnTensor::complex32_t> || 
                      std::is_same_v<T, OwnTensor::complex64_t> || 
                      std::is_same_v<T, OwnTensor::complex128_t>) {
             throw std::runtime_error("softplus_nodeops not implemented for complex types");
        } else {
            const T* x_data = x_val.data<T>();
            T* y_data = y.data<T>();
            int64_t n = x_val.numel();
            
            for (int64_t i = 0; i < n; ++i) {
                T val = x_data[i];
                if (val > T(threshold)) {
                    y_data[i] = val;  // For large x, softplus(x) ≈ x
                } else {
                    y_data[i] = std::log(T(1.0) + std::exp(val));
                }
            }
        }
    });

    auto n = std::make_shared<Node>(y, Op::Softplus, x->requires_grad(), "softplus");
    n->inputs = {x};
    if(x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}
std::shared_ptr<Node> gelu_nodeops(const std::shared_ptr<Node>& x){
    const float c1 = 0.7978845608f; 
    const float c2 = 0.044715f;
    Tensor x3 = x->value * x->value * x->value;
    Tensor u = (x->value + x3 * c2) * c1;
    Tensor y = x->value * (1.0f + OwnTensor::tanh(u)) * 0.5f;
    auto n = std::make_shared<Node>(y, Op::GELU, x->requires_grad(), "gelu");
    n->inputs={x};
    if(x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}
std::shared_ptr<Node> silu_nodeops(const std::shared_ptr<Node>& x){
    Tensor sig_x = 1.0f / (1.0f + OwnTensor::exp(x->value * -1.0f, ag::current_stream()));
    Tensor y = x->value * sig_x;
    auto n = std::make_shared<Node>(y, Op::SiLU, x->requires_grad(), "silu");
    n->inputs={x};
    if(x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}
std::shared_ptr<Node> mish_nodeops(const std::shared_ptr<Node>& x){
    Tensor sp = OwnTensor::log(1.0f + OwnTensor::exp(x->value, ag::current_stream()), ag::current_stream());
    Tensor y = x->value * OwnTensor::tanh(sp);
    auto n = std::make_shared<Node>(y, Op::Mish, x->requires_grad(), "mish");
    n->inputs = {x};
    if (x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}
std::shared_ptr<Node> leaky_relu_nodeops(const std::shared_ptr<Node>& x, float alpha){ 
    Tensor pos_part = (x->value + OwnTensor::abs(x->value, ag::current_stream())) * 0.5f;
    Tensor neg_part = (x->value - OwnTensor::abs(x->value, ag::current_stream())) * 0.5f;
    Tensor Y = pos_part + (neg_part * alpha);
    Tensor aT = Tensor::full(Shape{{1, 1}}, ag::options(x->value).with_req_grad(false), alpha);
    auto aC = make_tensor(aT, "alpha"); 
    auto n = std::make_shared<Node>(Y, Op::LeakyRelu, x->requires_grad(), "leakyrelu");
    n->inputs = {x, aC.node}; 
    if(x) x->child_grad_count++;
    if(aC.node) aC.node->child_grad_count++;
    ag::debug::on_node_created(n);  
    return n;
}
std::shared_ptr<Node> gaus_nodeops(const std::shared_ptr<Node>& x){
    Tensor x_squared = x->value * x->value;
    Tensor y = OwnTensor::exp(x_squared * -1.0f, ag::current_stream());
    auto n = std::make_shared<Node>(y, Op::Gaus, x->requires_grad(), "gaus");
    n->inputs={x};
    if(x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}

std::shared_ptr<Node> parcon_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = x->value * (2.0f - x->value);
    auto n = std::make_shared<Node>(y, Op::Parcon, x->requires_grad(), "parcon");
    n->inputs={x};
    if(x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}

std::shared_ptr<Node> lisht_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = x->value * OwnTensor::tanh(x->value);
    auto n = std::make_shared<Node>(y, Op::LiSHT, x->requires_grad(), "lisht"); 
    n->inputs={x};
    if(x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}

} // namespace detail
} // namespace ag
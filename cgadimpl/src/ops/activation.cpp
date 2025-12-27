// =====================
// file: cgadimpl/src/ops/activation.cpp
// =====================
#include "ad/ops/nodeops.hpp"
#include "ad/runtime/runtime.hpp"
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

std::shared_ptr<Node> relu_nodeops(const std::shared_ptr<Node>& x){
    const Tensor& X = x->value;
    Tensor Y = (X + OwnTensor::abs(X, ag::current_stream())) * 0.5f;
    auto n = std::make_shared<Node>(Y, Op::Relu, x->requires_grad(), "relu");
    n->inputs = {x};
    if(x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}

std::shared_ptr<Node> matmul_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b) {
    Tensor C = matmul(a->value, b->value);
    auto n = std::make_shared<Node>(C, Op::MatMul, (a->requires_grad() || b->requires_grad()), "matmul");
    n->inputs = {a, b};
    if(a) a->child_grad_count++;
    if(b) b->child_grad_count++;
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

std::shared_ptr<Node> reci_nodeops(const std::shared_ptr<Node>& a) {
    Tensor y = 1.0f / a->value;
    auto n = std::make_shared<Node>(y, Op::Reciprocal, a->requires_grad(),"reciprocal");
    n->inputs = {a};
    if(a) a->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}

std::shared_ptr<Node> relumask_nodeops(const std::shared_ptr<Node>& x) {
    const Tensor& xin = x->value;
    Tensor y = OwnTensor::Tensor::zeros(xin.shape(), ag::options(xin));
    if (xin.is_cpu()) {
        dispatch_by_dtype(xin.dtype(), [&](auto dummy){
            using T = decltype(dummy);
            const T* x_data = xin.data<T>();
            T* y_data = y.data<T>();
            for (int64_t i = 0; i < xin.numel(); ++i) {
                bool is_positive;
                if constexpr (std::is_same_v<T, OwnTensor::complex32_t> || 
                              std::is_same_v<T, OwnTensor::complex64_t> || 
                              std::is_same_v<T, OwnTensor::complex128_t>) {
                    is_positive = x_data[i].real() > 0;
                } else {
                    is_positive = x_data[i] > static_cast<T>(0);
                }
                if (is_positive) {
                    y_data[i] = static_cast<T>(1.0f);
                }
            }
        });
    } else {
        throw std::runtime_error("relumask_nodeops not implemented for CUDA yet.");
    }
    auto n = std::make_shared<Node>(y, Op::Relumask, x->requires_grad(), "relumask");
    n->inputs = {x};
    if(x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}

std::shared_ptr<Node> linear_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c) {
    const Tensor& input_X = a->value;
    const Tensor& weight_W = b->value; 
    const Tensor& bias_b = c->value;
    Tensor y = matmul(input_X, weight_W.t()) + bias_b;
    auto n = std::make_shared<Node>(y, Op::Linear, (a->requires_grad() || b->requires_grad() || c->requires_grad()), "linear");
    n->inputs = {a, b, c};
    if (a) a->child_grad_count++;
    if (b) b->child_grad_count++;
    if (c) c->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}

std::shared_ptr<Node> cosh_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = cosh(x->value);
    auto n=std::make_shared<Node>(y, Op::Cosh, x->requires_grad(), "cosh");
    n->inputs={x};
    if (x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}

std::shared_ptr<Node> sinh_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = sinh(x->value);
    auto n=std::make_shared<Node>(y, Op::Sinh, x->requires_grad(), "sinh");
    n->inputs={x};
    if (x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}

std::shared_ptr<Node> cos_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = cos(x->value);
    auto n=std::make_shared<Node>(y, Op::Cos, x->requires_grad(), "cos");
    n->inputs={x};
    if (x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}

std::shared_ptr<Node> sin_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = sin(x->value);
    auto n=std::make_shared<Node>(y, Op::Sin, x->requires_grad(), "sin");
    n->inputs={x};
    if(x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}

std::shared_ptr<Node> asinh_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = asinh(x->value);
    auto n=std::make_shared<Node>(y, Op::ASinh, x->requires_grad(), "asinh");
    n->inputs={x};
    if(x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}

std::shared_ptr<Node> acosh_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = acosh(x->value);
    auto n=std::make_shared<Node>(y, Op::ACosh, x->requires_grad(), "acosh");
    n->inputs={x};
    if(x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}

std::shared_ptr<Node> atanh_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = atanh(x->value);
    auto n=std::make_shared<Node>(y, Op::ATanh, x->requires_grad(), "atanh");
    n->inputs={x};
    if(x) x->child_grad_count++;
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

std::shared_ptr<Node> sqrt_nodeops(const std::shared_ptr<Node>& x) {
    Tensor y = OwnTensor::sqrt(x->value, ag::current_stream());
    auto n = std::make_shared<Node>(y, Op::Sqrt, x->requires_grad(), "sqrt");
    n->inputs = {x};
    if (x) x->child_grad_count++;
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

std::shared_ptr<Node> sum_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = OwnTensor::reduce_sum(x->value, {}, false);
    auto n = std::make_shared<Node>(y, Op::Sum, x->requires_grad(), "sum");
    n->inputs = {x};
    if(x) x->child_grad_count++;
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

std::shared_ptr<Node> mish_nodeops(const std::shared_ptr<Node>& x){
    Tensor sp = OwnTensor::log(1.0f + OwnTensor::exp(x->value, ag::current_stream()), ag::current_stream());
    Tensor y = x->value * OwnTensor::tanh(sp);
    auto n = std::make_shared<Node>(y, Op::Mish, x->requires_grad(), "mish");
    n->inputs = {x};
    if (x) x->child_grad_count++;
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

std::shared_ptr<Node> sigmoid_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = 1.0f / (1.0f + OwnTensor::exp(x->value * -1.0f, ag::current_stream()));
    auto n = std::make_shared<Node>(y, Op::Sigmoid, x->requires_grad(), "sigmoid"); 
    n->inputs={x}; 
    if(x) x->child_grad_count++;
    ag::debug::on_node_created(n);  
    return n;
}

std::shared_ptr<Node> softplus_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = OwnTensor::log(1.0f + OwnTensor::exp(x->value, ag::current_stream()), ag::current_stream());
    auto n = std::make_shared<Node>(y, Op::Softplus, x->requires_grad(), "softplus");
    n->inputs = {x};
    if(x) x->child_grad_count++;
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

std::shared_ptr<Node> laynor_nodeops(const std::shared_ptr<Node>& x){
    Tensor mean = OwnTensor::reduce_mean(x->value, {-1}, true);
    Tensor x_minus_mean = x->value - mean;
    Tensor variance = OwnTensor::reduce_mean(x_minus_mean * x_minus_mean, {-1}, true);
    Tensor y = x_minus_mean / OwnTensor::sqrt(variance + 1e-5f, ag::current_stream());
    auto n = std::make_shared<Node>(y, Op::LayerNorm, x->requires_grad(), "layernorm");
    n->tape.push_back(std::make_shared<Tensor>(variance));
    n->tape.push_back(std::make_shared<Tensor>(mean));
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

std::shared_ptr<Node> abs_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = OwnTensor::abs(x->value, ag::current_stream());
    auto n = std::make_shared<Node>(y, Op::Abs, x->requires_grad(), "abs");
    n->inputs={x};
    if(x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}

std::shared_ptr<Node> dyntanh_nodeops(const std::shared_ptr<Node>& x, float& a_val, float& b_val, float& g_val){
    std::shared_ptr<Node> A = std::make_shared<Node>(Tensor::full(Shape{{1}}, TensorOptions().with_req_grad(true), a_val), Op::Leaf, "dyn_a");
    std::shared_ptr<Node> B = std::make_shared<Node>(Tensor::full(Shape{{1}}, TensorOptions().with_req_grad(true), b_val), Op::Leaf, "dyn_b");
    std::shared_ptr<Node> G = std::make_shared<Node>(Tensor::full(Shape{{1}}, TensorOptions().with_req_grad(true), g_val), Op::Leaf, "dyn_g");
    Tensor h = x->value * A->value;
    Tensor y = OwnTensor::tanh(h) * G->value + B->value;
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

} // namespace detail
} // namespace ag
// =====================
// file: src/ops.cpp
// =====================

#include <iostream>
#include "ad/ops.hpp"
#include "ad/graph.hpp"
#include "ad/debug.hpp"
#include "ad/kernels_api.hpp"
#include <math.h>
#include <iterator>
#include "ad/checkpoint.hpp"

namespace ag {


    Value add(const Value& a, const Value& b){ 
        Tensor y = a.val() + b.val(); 
        auto n = std::make_shared<Node>(y, a.node->requires_grad || b.node->requires_grad, Op::Add, "+"); 
        n->inputs = {a.node, b.node}; 
        ag::debug::on_node_created(n); 
        return Value(n); 
    }

    Value sub(const Value& a, const Value& b){ 
        Tensor y = a.val() - b.val(); 
        auto n = std::make_shared<Node>(y, a.node->requires_grad || b.node->requires_grad, Op::Sub, "-"); 
        n->inputs = {a.node, b.node}; 
        ag::debug::on_node_created(n); 
        return Value(n); 
    }



    Value mul(const Value& a, const Value& b){ 
        Tensor y = a.val() * b.val(); 
        auto n = std::make_shared<Node>(y, a.node->requires_grad || b.node->requires_grad, Op::Mul, "*"); 
        n->inputs = {a.node, b.node}; 
        ag::debug::on_node_created(n); 
        return Value(n); 
    }

    Value flomul(const Value& a, float b){ 
        auto c = constant(b*Tensor::ones_like(a.val()));
        Tensor y = a.val() * c.val(); 
        auto n = std::make_shared<Node>(y, a.node->requires_grad || c.node->requires_grad, Op::Mul, "*"); 
        n->inputs = {a.node, c.node}; 
        ag::debug::on_node_created(n); 
        return Value(n); 
    }

    Value relu(const Value& x){ 
        const Tensor& xin = x.val();
        Tensor y = Tensor::zeros_like(xin);

        auto* fn = ag::kernels::cpu().relu;
        if (!fn) throw std::runtime_error("No CPU ReLU kernel registered");
        fn(xin.data(), y.data(), static_cast<int64_t>(xin.numel()));

        auto n = std::make_shared<Node>(y, x.node->requires_grad, Op::Relu, "relu");
        n->inputs = { x.node };
        return Value(std::move(n));
    }





    Value matmul(const Value& a, const Value& b){ 
         const Tensor& A = a.val();
         const Tensor& B = b.val();

         auto [M,K]  = A.shape();
         auto [K2,N] = B.shape();
         if (K != K2) throw std::runtime_error("matmul: inner dims mismatch");

         Tensor C({M,N});

         auto* fn = ag::kernels::cpu().matmul;
         if (!fn) throw std::runtime_error("No CPU MatMul kernel registered");
         fn(A.data(), B.data(), C.data(), M, K, N);

         auto n = std::make_shared<Node>(C,
             (a.node->requires_grad || b.node->requires_grad),
             Op::MatMul, "matmul");
         n->inputs = { a.node, b.node };
         return Value(std::move(n));
    }

    Value fmab(const Value& a, const Value& b, const Value& c){ 
        Tensor y = Tensor::matmul(a.val(), b.val())+c.val(); 
        auto n = std::make_shared<Node>(y, a.node->requires_grad || b.node->requires_grad || c.node->requires_grad, Op::FMA, "fmab"); 
        n->inputs = {a.node, b.node, c.node}; ag::debug::on_node_created(n); 
        return Value(n); 
    }


    Value attention(const Value& a, const Value& b, const Value& c, const Value& d){ 
    Tensor q = Tensor::matmul(a.val(), b.val()); 
    Tensor k = Tensor::matmul(a.val(), c.val()); 
    Tensor v = Tensor::matmul(a.val(), d.val());
    Tensor g = Tensor::matmul(q, Tensor::transpose(k)*(1.f/sqrt(float(k.cols())))) ;
    Tensor s = Tensor::softmax_row(g);
    Tensor y = Tensor::matmul(s, v);
    auto n = std::make_shared<Node>(y, a.node->requires_grad || b.node->requires_grad || c.node->requires_grad || d.node->requires_grad, Op::Attention, "attention"); 
    n->inputs = {a.node, b.node, c.node, d.node};
    n->tape.resize(4);
    n->tape={std::make_shared<Tensor>(q), std::make_shared<Tensor>(k), std::make_shared<Tensor>(v), std::make_shared<Tensor>(s)};
    ag::debug::on_node_created(n); 
    return Value(n); 
    }


    Value alibiatt(const Value& a, const Value& b, const Value& c, const Value& d, float m) { 
    Tensor q = Tensor::matmul(a.val(), b.val()); 
    Tensor k = Tensor::matmul(a.val(), c.val()); 
    Tensor v = Tensor::matmul(a.val(), d.val());
    
    Tensor logits = Tensor::matmul(q, Tensor::transpose(k) * (1.f / sqrt(float(k.cols()))));
    Tensor bias   = Tensor::alibi(logits.rows(), logits.cols(), m);
    Tensor g      = logits + bias;

    Tensor s = Tensor::softmax_row(g);
    Tensor y = Tensor::matmul(s, v);

    auto n = std::make_shared<Node>(
        y, a.node->requires_grad || b.node->requires_grad || c.node->requires_grad || d.node->requires_grad,
        Op::AlibiAttention, "alibiattention"
    ); 
    n->inputs = {a.node, b.node, c.node, d.node};
    n->tape.resize(4);
    n->tape   = {std::make_shared<Tensor>(q), std::make_shared<Tensor>(k), 
                 std::make_shared<Tensor>(v), std::make_shared<Tensor>(s)};
    ag::debug::on_node_created(n); 
    return Value(n); 
}



    Value swiglu(const Value& x, const Value& a, const Value& b, const Value& c, const Value& d){ 
    Tensor y = Tensor::matmul(x.val(), Tensor::transpose(a.val()))+b.val(); 
    debug::print_tensor("y",y);
    Tensor q = y*Tensor::sigmoid(y); 
    Tensor w = q*(Tensor::matmul(x.val(), Tensor::transpose(c.val())) + d.val());
    auto n=std::make_shared<Node>(w, x.node->requires_grad || a.node->requires_grad || b.node->requires_grad || c.node->requires_grad || d.node->requires_grad, Op::SWIGLU, "swiglu"); 
    n->inputs={x.node, a.node, b.node, c.node, d.node};
    ag::debug::on_node_created(n); 
    return Value(n);
    }







    Value sum(const Value& x){ 
        Tensor y = Tensor::sum_all(x.val()); 
        auto n = std::make_shared<Node>(y, x.node->requires_grad, Op::Sum, "sum"); 
        n->inputs = {x.node}; 
        ag::debug::on_node_created(n);  
        return Value(n); 
    }

    Value transpose(const Value& x){ 
        Tensor y = Tensor::transpose(x.val()); 
        auto n=std::make_shared<Node>(y, x.node->requires_grad, Op::Transpose, "exp"); 
        n->inputs={x.node}; 
        ag::debug::on_node_created(n);  
        return Value(n);
    }

    Value exp(const Value& x){ 
        Tensor y = Tensor::exp(x.val()); 
        auto n=std::make_shared<Node>(y, x.node->requires_grad, Op::Exp, "exp"); 
        n->inputs={x.node}; 
        ag::debug::on_node_created(n);  
        return Value(n);
    }
    
    Value log(const Value& x){ 
        Tensor y = Tensor::log(x.val()); 
        auto n=std::make_shared<Node>(y, x.node->requires_grad, Op::Log, "log"); 
        n->inputs={x.node}; 
        ag::debug::on_node_created(n);  
        return Value(n);
    }


    Value mish(const Value& x){ 
        Tensor y = x.val() * Tensor::tanh( Tensor::softplus(x.val()) ); 
        auto n=std::make_shared<Node>(y, x.node->requires_grad, Op::Mish, "mish"); 
        n->inputs={x.node}; 
        ag::debug::on_node_created(n);  
        return Value(n);
    }
    
    Value tanh(const Value& x){ 
        Tensor y = Tensor::tanh(x.val()); 
        auto n=std::make_shared<Node>(y, x.node->requires_grad, Op::Tanh, "tanh"); 
        n->inputs={x.node}; 
        ag::debug::on_node_created(n);  
        return Value(n);
    }
    
    Value sigmoid(const Value& x){ 
        Tensor y = Tensor::sigmoid(x.val()); 
        auto n=std::make_shared<Node>(y, x.node->requires_grad, Op::Sigmoid, "sigmoid"); 
        n->inputs={x.node}; 
        ag::debug::on_node_created(n);  
        return Value(n);
    }
    
    Value softplus(const Value& x){ 
        Tensor y = Tensor::softplus(x.val()); 
        auto n=std::make_shared<Node>(y, x.node->requires_grad, Op::Softplus, "softplus"); 
        n->inputs={x.node}; 
        ag::debug::on_node_created(n);  
        return Value(n);
    }

    Value gaus(const Value& x){ 
        Tensor y = Tensor::exp(-1*x.val()*x.val()); 
        auto n=std::make_shared<Node>(y, x.node->requires_grad, Op::Gaus, "gaus"); 
        n->inputs={x.node}; 
        ag::debug::on_node_created(n);  
        return Value(n);
    }
    
    Value gelu(const Value& x){ 
        Tensor y = Tensor::gelu_tanh(x.val()); 
        auto n=std::make_shared<Node>(y, x.node->requires_grad, Op::GELU, "gelu"); 
        n->inputs={x.node}; 
        ag::debug::on_node_created(n);  
        return Value(n);
    }



    Value gcu(const Value& x){ 
        Tensor y = x.val() * Tensor::cos(x.val());
        auto n=std::make_shared<Node>(y, x.node->requires_grad, Op::GCU, "gcu"); 
        n->inputs={x.node}; 
        ag::debug::on_node_created(n);  
        return Value(n);
    }
    
    Value silu(const Value& x){ 
        Tensor y = Tensor::sigmoid(x.val()); 
        y = y * x.val(); 
        auto n=std::make_shared<Node>(y, x.node->requires_grad, Op::SiLU, "silu"); 
        n->inputs={x.node}; 
        ag::debug::on_node_created(n);  
        return Value(n);
    }

    Value parcon(const Value& x){ 
        Tensor y = x.val()*(2*Tensor::ones_like(x.val())-x.val()); 

        auto n=std::make_shared<Node>(y, x.node->requires_grad, Op::Parcon, "parcon"); 
        n->inputs={x.node}; 
        ag::debug::on_node_created(n);  
        return Value(n);
    }

    Value lisht(const Value& x){ 
        Tensor y = x.val()*Tensor::tanh(x.val()); 

        auto n=std::make_shared<Node>(y, x.node->requires_grad, Op::Parcon, "parcon"); 
        n->inputs={x.node}; 
        ag::debug::on_node_created(n);  
        return Value(n);
    }
    
    Value leaky_relu(const Value& x, float alpha){ 
        Tensor y = Tensor::leaky_relu(x.val(), alpha); 
        Tensor aT(1,1); aT(0,0)=alpha; auto aC = constant(aT, "alpha"); 
        auto n=std::make_shared<Node>(y, x.node->requires_grad, Op::LeakyRelu, "leakyrelu");
        n->inputs={x.node, aC.node}; 
        ag::debug::on_node_created(n);  
        return Value(n);
    }


    Value rowsum(const Value& x){ 
        Tensor y = Tensor::row_sum(x.val()); 
        auto n=std::make_shared<Node>(y, x.node->requires_grad, Op::RowSum, "rowsum"); 
        n->inputs={x.node}; 
        ag::debug::on_node_created(n);  
        return Value(n);
    }
    
    Value rowmax(const Value& x){ 
        Tensor y = Tensor::row_max(x.val()); 
        auto n=std::make_shared<Node>(y, x.node->requires_grad, Op::RowMax, "rowmax"); 
        n->inputs={x.node}; 
        ag::debug::on_node_created(n);  
        return Value(n);
    }

    Value rms(const Value& x){ 
Tensor z = Tensor::row_sum(x.val()*x.val()) * (1.f/x.val().cols());
Tensor q = Tensor::sqrt(z + 1e-8f);
Tensor y = x.val() / q;

auto n = std::make_shared<Node>(y, x.node->requires_grad, Op::RMSNorm, "rmsnorm");
n->tape.resize(2);
n->tape[0] = std::make_shared<Tensor>(q); // denominator
n->tape[1] = std::make_shared<Tensor>(y);   // normalized output
n->inputs = {x.node};
return Value(n);
    }

    Value realrms(const Value& x, float g){ 
Tensor z = Tensor::row_sum(x.val()*x.val()) * (1.f/x.val().cols());
Tensor q = Tensor::sqrt(z + 1e-8f);
Tensor y = (x.val()) / q;
        Value G = param(g*Tensor::ones_like(y), "g");

auto n = std::make_shared<Node>(y*g, x.node->requires_grad || G.node->requires_grad, Op::RealRMSNorm, "realrmsnorm");
n->tape.resize(2);
n->tape[0] = std::make_shared<Tensor>(q); // denominator
n->tape[1] = std::make_shared<Tensor>(y);   // normalized output
n->inputs = {x.node, G.node};
return Value(n);
    }

    Value laynor(const Value& x){ 
        Tensor y = Tensor::row_sum(x.val())*(1.f/x.val().cols()); 
      //  std::cout<<"q      "<<y<<std::endl;
        Tensor vrc = Tensor::row_sum(((x.val() )- y)*((x.val() )- y))*(1.f/x.val().cols());
      //  std::cout<<"q      "<<vrc<<std::endl;
        Tensor q = ((x.val() )- y)/(Tensor::sqrt(vrc)+0.01);
        
        auto n=std::make_shared<Node>(q, x.node->requires_grad, Op::LayerNorm, "layernorm");
      //  debug::print_tensor("q",q);
        n->tape.resize(2);
        n->tape[0] = std::make_shared<Tensor>(vrc);
        n->tape[1] = std::make_shared<Tensor>(y);

        n->inputs={x.node}; 
        ag::debug::on_node_created(n);  
        return Value(n);
    }

    Value relaynor(const Value& x, float b, float g){ 
        Tensor y = Tensor::row_sum(x.val())*(1.f/x.val().cols()); 
      //  std::cout<<"q      "<<y<<std::endl;
        Tensor vrc = Tensor::row_sum(((x.val() )- y)*((x.val() )- y))*(1.f/x.val().cols());
      //  std::cout<<"q      "<<vrc<<std::endl;
        Tensor q = ((((x.val() )- y)/(Tensor::sqrt(vrc)+0.01)))   ;
        Tensor qg = (q*g) + b;

        Value B = param(b*Tensor::ones_like(qg), "b");
        Value G = param(g*Tensor::ones_like(qg), "g");
        
        auto n=std::make_shared<Node>(q, x.node->requires_grad || B.node->requires_grad||G.node->requires_grad, Op::RealLayerNorm, "reallayernorm");
      //  debug::print_tensor("q",q);
        n->tape.resize(3);
        n->tape[0] = std::make_shared<Tensor>(vrc);
        n->tape[1] = std::make_shared<Tensor>(y);
        n->tape[2] = std::make_shared<Tensor>(q);

        n->inputs={x.node}; 
        ag::debug::on_node_created(n);  
        return Value(n);
    }
    
    Value mean_all(const Value& x){ 
        Tensor y = Tensor::mean_all(x.val()); 
        auto n=std::make_shared<Node>(y, x.node->requires_grad, Op::MeanAll, "meanall"); 
        n->inputs={x.node}; 
        ag::debug::on_node_created(n);  
        return Value(n);
    }

    Value dyntanh(const Value& x, float a, float b, float g){ 
        Tensor h = x.val()*a;
        Tensor y = Tensor::tanh(h)*g + b; 
        Value A = param(a*Tensor::ones_like(x.val()), "a");
        Value B = param(b*Tensor::ones_like(x.val()), "b");
        Value G = param(g*Tensor::ones_like(x.val()), "g");
        auto n=std::make_shared<Node>(y, x.node->requires_grad|| A.node->requires_grad|| B.node->requires_grad||G.node->requires_grad, Op::MeanAll, "meanall"); 
        n->inputs={x.node, A.node, B.node, G.node}; 
        n->tape.push_back(std::make_shared<Tensor>(h));
        ag::debug::on_node_created(n);  
        return Value(n);
    }
    
    Value softmax_row(const Value& z){ 
        Tensor y = Tensor::softmax_row(z.val()); 
        auto n=std::make_shared<Node>(y, z.node->requires_grad, Op::SoftmaxRow, "softmax_row"); 
        n->inputs={z.node}; 
        ag::debug::on_node_created(n);  
        return Value(n);
    }
    
    Value logsumexp_row(const Value& z){ 
        Tensor y = Tensor::logsumexp_row(z.val()); 
        auto n=std::make_shared<Node>(y, z.node->requires_grad, Op::LogSumExpRow, "logsumexp_row"); 
        n->inputs={z.node}; 
        ag::debug::on_node_created(n);  
        return Value(n);
    }


    Value mambassm(const Value& z, const Value& a, const Value& b, const Value& c, const Value& d){ 

        if (z.node->tape.size()==0) {

                    Tensor w = Tensor::matmul(z.val(), b.val()); 
                    Tensor q = Tensor::matmul(w, c.val());
                    Tensor y = (z.val()* d.val())+q;
                    auto W = param(w, "w");
        auto n=std::make_shared<Node>(y, W.node->requires_grad || z.node->requires_grad || a.node->requires_grad || b.node->requires_grad || c.node->requires_grad || d.node->requires_grad, Op::LogSumExpRow, "logsumexp_row"); 
        n->inputs={z.node, a.node, b.node, c.node, d.node, W.node}; 
            z.node->tape.push_back(std::make_shared<Tensor>(w));
            z.node->inputs.push_back(W.node);
                    ag::debug::on_node_created(n);  
                    std::cout<<"Initialized SSM state"<<std::endl;
return Value(n);
        }
        else
        {

Tensor w = Tensor::matmul(z.val(), b.val())+(z.node->inputs.back()->value); 
                    Tensor q = Tensor::matmul(w, c.val());
                    Tensor y = (z.val()* d.val())+q;
                    auto W = param(w, "w");
        auto n=std::make_shared<Node>(y,  W.node->requires_grad || z.node->requires_grad || a.node->requires_grad || b.node->requires_grad || c.node->requires_grad || d.node->requires_grad, Op::LogSumExpRow, "logsumexp_row"); 
        n->inputs={z.node, a.node, b.node, c.node, d.node, W.node}; 
        z.node->tape.push_back(std::make_shared<Tensor>(w));
            z.node->inputs.push_back(W.node);
                    ag::debug::on_node_created(n);  
                    std::cout<<"SSM step"<<std::endl;
return Value(n);
        }

        
    }


    Value cross_entropy_with_logits(const Value& logits, const Value& onehot){
    // Stable CE = mean( -sum(onehot * (logits - logsumexp_row(logits))) )
        Tensor Z = logits.val();
        Tensor Y = onehot.val();
        Tensor LSE = Tensor::logsumexp_row(Z); // [B,1]
        Tensor log_sm = Z - LSE; // [B,C]
        Tensor prod = Y * log_sm; // [B,C]
        Tensor rs = Tensor::row_sum(prod); // [B,1]
        Tensor s = Tensor::sum_all(rs); // [1,1]
        Tensor out = Tensor::mean_all(rs * Tensor::ones_like(rs)); // mean over B (same as s/B)
        // We'll compute exact mean: s / B
        Tensor mean(1,1); mean(0,0) = s(0,0) / float(Z.rows());
        Tensor loss = Tensor::zeros(1,1); loss(0,0) = -mean(0,0);
        auto n = std::make_shared<Node>(loss, logits.node->requires_grad || onehot.node->requires_grad, Op::CeWithLogits, "ce_with_logits");
        n->inputs = {logits.node, onehot.node};
        ag::debug::on_node_created(n);  
        return Value(n);
    }


    Value kldivergence(const Value& logits, const Value& onehot){
    // Stable CE = mean( -sum(onehot * (logits - logsumexp_row(logits))) )
        Tensor Z = logits.val();
        Tensor Y = onehot.val();
        Tensor X = Tensor::log(Y + Tensor::ones_like(Y)*1e-10f); // add small value to avoid log(0)
        Tensor LSE = Tensor::logsumexp_row(Z); // [B,1]
        Tensor log_sm = X- Z + LSE; // [B,C]
        Tensor prod = Y * log_sm; // [B,C]
        Tensor rs = Tensor::row_sum(prod); // [B,1]
        Tensor s = Tensor::sum_all(rs); // [1,1]
        Tensor out = Tensor::mean_all(rs * Tensor::ones_like(rs)); // mean over B (same as s/B)
        // We'll compute exact mean: s / B
        Tensor mean(1,1); mean(0,0) = s(0,0) / float(Z.rows());
        Tensor loss = Tensor::zeros(1,1); loss(0,0) = -mean(0,0);
        auto n = std::make_shared<Node>(loss, logits.node->requires_grad || onehot.node->requires_grad, Op::KLDivergence, "kldivergence");
        n->inputs = {logits.node, onehot.node};
        ag::debug::on_node_created(n);  
        return Value(n);
    }

    Value mse_loss(const Value& pred, const Value& target) {
    Tensor diff = pred.val() - target.val();
    Tensor sq   = diff * diff;               // elementwise
    Tensor s    = Tensor::sum_all(sq);                   // scalar [1,1]
    int B = pred.shape().first, C = pred.shape().second;
    Tensor scale = Tensor::ones(1,1);
    scale(0,0) = 1.0f / float(B * C);
    Tensor loss = s * scale;                 // broadcast scalar
    auto n = std::make_shared<Node>(loss, pred.node->requires_grad || target.node->requires_grad, Op::MSELoss, "mseloss");
    n->inputs = {pred.node, target.node};
        ag::debug::on_node_created(n);  
    return Value(n);                 // broadcast scalar
}


    Value mae_loss(const Value& pred, const Value& target) {
    Tensor diff = pred.val() - target.val();
    Tensor sq   = Tensor::abs(diff);               // elementwise
    Tensor s    = Tensor::sum_all(sq);                   // scalar [1,1]
    int B = pred.shape().first, C = pred.shape().second;
    Tensor scale = Tensor::ones(1,1);
    scale(0,0) = 1.0f / float(B * C);
    Tensor loss = s * scale;                 // broadcast scalar
    auto n = std::make_shared<Node>(loss, pred.node->requires_grad || target.node->requires_grad, Op::MAELoss, "maeloss");
    n->inputs = {pred.node, target.node};
        ag::debug::on_node_created(n);  
    return Value(n);                 // broadcast scalar
}

Tensor forward_eval_node(const std::shared_ptr<Node> &node) {
    if (!node) throw std::runtime_error("forward_eval_node: null node");

    switch (node->op) {
        case Op::Add: {
            const Tensor &A = node->inputs[0]->value;
            const Tensor &B = node->inputs[1]->value;
            return A + B;
        }
        case Op::Sub: {
            const Tensor &A = node->inputs[0]->value;
            const Tensor &B = node->inputs[1]->value;
            return A - B;
        }
        case Op::Mul: {
            const Tensor &A = node->inputs[0]->value;
            const Tensor &B = node->inputs[1]->value;
            return A * B;
        }
        case Op::MatMul: {
            const Tensor &A = node->inputs[0]->value;
            const Tensor &B = node->inputs[1]->value;
            return Tensor::matmul(A, B);
        }
        case Op::Relu: {
            const Tensor &X = node->inputs[0]->value;
            return Tensor::relu(X);
        }
        case Op::Sigmoid: {
            const Tensor &X = node->inputs[0]->value;
            return Tensor::sigmoid(X);
        }
        case Op::Tanh: {
            const Tensor &X = node->inputs[0]->value;
            return Tensor::tanh(X);
        }
        case Op::Exp: {
            const Tensor &X = node->inputs[0]->value;
            return Tensor::exp(X);
        }
        case Op::Log: {
            const Tensor &X = node->inputs[0]->value;
            return Tensor::log(X);
        }
        case Op::AlibiAttention: {
            const Tensor &a = node->inputs[0]->value;
            const Tensor &b = node->inputs[1]->value;
            const Tensor &c = node->inputs[2]->value;
            const Tensor &d = node->inputs[3]->value;

            Tensor q = Tensor::matmul(a, b);
            Tensor k = Tensor::matmul(a, c);
            Tensor v = Tensor::matmul(a, d);

            Tensor logits = Tensor::matmul(q, Tensor::transpose(k) * (1.f / sqrt(float(k.cols()))));
            Tensor bias   = Tensor::alibi(logits.rows(), logits.cols(), /*m*/128);
            Tensor g      = logits + bias;
            Tensor s      = Tensor::softmax_row(g);
            Tensor y      = Tensor::matmul(s, v);
            return y;
        }
        case Op::Leaf:
            return node->value;
        default:
            if (!node->tape.empty()) {
                return *(node->tape.back());
            }
            throw std::runtime_error("forward_eval_node: unsupported op for recompute");
    }
}

// ------------------------------------------------------------
// Small adapter so checkpoint.cpp (which uses Node*) can link.
// ------------------------------------------------------------
Tensor forward_eval_node(Node* node) {
    // Non-owning shared_ptr wrapper (no deletion)
    return forward_eval_node(std::shared_ptr<Node>(node, [](Node*){}));
}

// ------------------------------------------------------------
// checkpoint() — mark a node for checkpointing
// ------------------------------------------------------------
Value checkpoint(const Value &v, const CheckpointOptions &opts) {
    if (!v.node) return v;
    ag::checkpoint_impl::mark_node_checkpoint(v.node, opts);
    return v;
}

} // namespace ag
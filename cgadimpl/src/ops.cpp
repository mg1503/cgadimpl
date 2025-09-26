// =====================
// file: src/ops.cpp
// =====================

#include <iostream>
#include "ad/ops.hpp"
#include "ad/debug.hpp"
#include <math.h>


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

    Value relu(const Value& x){ 
        Tensor y = Tensor::relu(x.val()); 
        auto n = std::make_shared<Node>(y, x.node->requires_grad, Op::Relu, "relu"); 
        n->inputs = {x.node}; 
        ag::debug::on_node_created(n);  
        return Value(n); 
    }

    Value matmul(const Value& a, const Value& b){ 
        Tensor y = Tensor::matmul(a.val(), b.val()); 
        auto n = std::make_shared<Node>(y, a.node->requires_grad || b.node->requires_grad, Op::MatMul, "matmul"); 
        n->inputs = {a.node, b.node}; ag::debug::on_node_created(n); 
        return Value(n); 
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
    Tensor g = Tensor::matmul(q, Tensor::transpose(k)) *(1/sqrt(float(k.cols())));
    Tensor s = Tensor::softmax_row(g);
    Tensor y = Tensor::matmul(s, v);
    auto n = std::make_shared<Node>(y, a.node->requires_grad || b.node->requires_grad || c.node->requires_grad || d.node->requires_grad, Op::Attention, "attention"); 
    n->inputs = {a.node, b.node, c.node, d.node};
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
    
    Value gelu(const Value& x){ 
        Tensor y = Tensor::gelu_tanh(x.val()); 
        auto n=std::make_shared<Node>(y, x.node->requires_grad, Op::GELU, "gelu"); 
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
    
    Value mean_all(const Value& x){ 
        Tensor y = Tensor::mean_all(x.val()); 
        auto n=std::make_shared<Node>(y, x.node->requires_grad, Op::MeanAll, "meanall"); 
        n->inputs={x.node}; 
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


} // namespace ag
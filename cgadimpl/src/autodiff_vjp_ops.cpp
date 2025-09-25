#include "ad/autodiff_ops.hpp"

namespace ag {
namespace {

// helper: reduce a gradient to a parent's shape (broadcast-aware)
inline Tensor rt(const Tensor& g, const Tensor& like){ return Tensor::reduce_to(g, like); }

// ----- elementwise binary -----
void vjp_Add(Node* n, const Tensor& gy){
    Node* A = n->inputs[0].get(); Node* B = n->inputs[1].get();
    if (A->requires_grad) A->grad.add_( rt(gy, A->value) );
    if (B->requires_grad) B->grad.add_( rt(gy, B->value) );
}
void vjp_Sub(Node* n, const Tensor& gy){
    Node* A = n->inputs[0].get(); Node* B = n->inputs[1].get();
    if (A->requires_grad) A->grad.add_( rt(gy, A->value) );
    if (B->requires_grad) B->grad.add_( rt(-gy, B->value) );
}
void vjp_Mul(Node* n, const Tensor& gy){
    Node* A = n->inputs[0].get(); Node* B = n->inputs[1].get();
    if (A->requires_grad) A->grad.add_( rt( gy * B->value, A->value) );
    if (B->requires_grad) B->grad.add_( rt( gy * A->value, B->value) );
}

// ----- elementwise trinary -----
void vjp_FMA(Node* n, const Tensor& gy){
    Node* A = n->inputs[0].get(); Node* B = n->inputs[1].get(); Node* C = n->inputs[2].get();


        if (A->requires_grad) A->grad.add_( Tensor::matmul(gy, Tensor::transpose(B->value)) );
    if (B->requires_grad) B->grad.add_( Tensor::matmul(Tensor::transpose(A->value), gy) );
    if (C->requires_grad) C->grad.add_( rt(gy, C->value) );
}


// ----- unary activations -----
void vjp_Relu(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad) return;
    int R=X->value.rows(), C=X->value.cols();
    Tensor g(R,C);
    for (int i=0;i<R;++i) for (int j=0;j<C;++j)
        g(i,j) = (n->value(i,j) > 0.f) ? gy(i,j) : 0.f;
    X->grad.add_( g );
}
void vjp_Exp(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (X->requires_grad) X->grad.add_( rt( gy * Tensor::exp(X->value), X->value) );
}
void vjp_Log(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (X->requires_grad) X->grad.add_( rt( gy / X->value, X->value) );
}
void vjp_Tanh(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad) return;
    Tensor th = n->value, one = Tensor::ones_like(th);
    X->grad.add_( rt( gy * (one - th*th), X->value) );
}
void vjp_Sigmoid(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get(); if (!X->requires_grad) return;
    Tensor s = Tensor::sigmoid(X->value);
    X->grad.add_( rt( gy * ( s * (Tensor::ones_like(s)-s) ), X->value) );
}
void vjp_Softplus(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get(); if (!X->requires_grad) return;
    X->grad.add_( rt( gy * Tensor::sigmoid(X->value), X->value) );
}
void vjp_SiLU(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get(); if (!X->requires_grad) return;
    Tensor s = Tensor::sigmoid(X->value);
    X->grad.add_( rt( gy * ( s + X->value * ( s * (Tensor::ones_like(s)-s) ) ), X->value) );
}
void vjp_GELU(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get(); if (!X->requires_grad) return;
    constexpr float c = 0.79788456080286535588f; // sqrt(2/pi)
    int R=X->value.rows(), C=X->value.cols();
    Tensor x=X->value,u(R,C),dudx(R,C);
    for(int i=0;i<R;++i) for(int j=0;j<C;++j){
        float z=x(i,j);
        u(i,j)=c*(z+0.044715f*z*z*z);
        dudx(i,j)=c*(1.f+0.134145f*z*z);
    }
    Tensor th=Tensor::tanh(u), one=Tensor::ones_like(th);
    Tensor dgelu=(one+th)*0.5f + (x * ((one - th*th) * dudx))*0.5f;
    X->grad.add_( rt( gy * dgelu, X->value) );
}
void vjp_LeakyRelu(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get(); Node* A = n->inputs[1].get();
    if (!X->requires_grad) return;
    float a = A->value(0,0);
    int R=X->value.rows(), C=X->value.cols();
    Tensor g(R,C);
    for(int i=0;i<R;++i) for(int j=0;j<C;++j){
        float z=X->value(i,j);
        g(i,j)= gy(i,j) * (z>0.f ? 1.f : a);
    }
    X->grad.add_( g );
}

// ----- matmul -----
void vjp_MatMul(Node* n, const Tensor& gy){
    Node* A = n->inputs[0].get(); Node* B = n->inputs[1].get();
    if (A->requires_grad) A->grad.add_( Tensor::matmul(gy, Tensor::transpose(B->value)) );
    if (B->requires_grad) B->grad.add_( Tensor::matmul(Tensor::transpose(A->value), gy) );
}

// ----- reductions -----
void vjp_Sum(Node* n, const Tensor& gy){
    Node* X=n->inputs[0].get(); if(!X->requires_grad) return;
    float s = gy(0,0);
    X->grad.add_( Tensor::ones_like(X->value) * s );
}
void vjp_RowSum(Node* n, const Tensor& gy){
    Node* X=n->inputs[0].get(); if(!X->requires_grad) return;
    // gy [B,1] broadcast across columns
    Tensor g = gy * Tensor::ones_like(X->value);
    X->grad.add_( g );
}
void vjp_RowMax(Node* n, const Tensor& gy){
    Node* X=n->inputs[0].get(); if(!X->requires_grad) return;
    int R=X->value.rows(), C=X->value.cols();
    Tensor m = Tensor::row_max(X->value);
    Tensor g(R,C);
    for(int i=0;i<R;++i) for(int j=0;j<C;++j)
        g(i,j) = (X->value(i,j)==m(i,0)) ? gy(i,0) : 0.f;
    X->grad.add_( g );
}
void vjp_MeanAll(Node* n, const Tensor& gy){
    Node* X=n->inputs[0].get(); if(!X->requires_grad) return;
    float scale = gy(0,0) / float(X->value.rows()*X->value.cols());
    X->grad.add_( Tensor::ones_like(X->value) * scale );
}

// ----- softmax / losses -----
void vjp_SoftmaxRow(Node* n, const Tensor& gy){
    Node* Z = n->inputs[0].get(); if(!Z->requires_grad) return;
    Tensor y = n->value; // softmax(Z)
    Tensor dot = Tensor::row_sum( y * gy ); // [B,1]
    Tensor g = y * (gy - dot);
    Z->grad.add_( g );
}
void vjp_LogSumExpRow(Node* n, const Tensor& gy){
    Node* Z = n->inputs[0].get(); if(!Z->requires_grad) return;
    Tensor y = Tensor::softmax_row(Z->value);
    Z->grad.add_( y * gy ); // gy [B,1] broadcast
}
void vjp_CeWithLogits(Node* n, const Tensor& gy /*unused: scalar gy*/){
    Node* Z = n->inputs[0].get();
    Node* Y = n->inputs[1].get();
    int B = Z->value.rows();
    Tensor sm = Tensor::softmax_row(Z->value);
    Tensor gZ = (sm - Y->value) * (1.0f / float(B));
    if (Z->requires_grad) Z->grad.add_( gZ );
    if (Y->requires_grad) {
        Tensor lse = Tensor::logsumexp_row(Z->value);
        Tensor lsm = Z->value - lse;
        Tensor gY  = lsm * (-1.0f / float(B));
        Y->grad.add_( gY );
    }
}



void vjp_KLDivergence(Node* n, const Tensor& gy /*unused: scalar gy*/){
    Node* Z = n->inputs[0].get();
    Node* Y = n->inputs[1].get();
    int B = Z->value.rows();
    Tensor sm = Tensor::softmax_row(Z->value);
    Tensor gZ = (sm - Y->value) * (1.0f / float(B));
    if (Z->requires_grad) Z->grad.add_( gZ );
    if (Y->requires_grad) {
        Tensor lse = Tensor::logsumexp_row(Z->value);
        Tensor lsm = Z->value - lse;
        Tensor gY = (Tensor::log(Y->value) + Tensor::ones_like(Y->value) - lsm) * (1.0f / float(B));
        Y->grad.add_( gY );
    }
}
void vjp_Leaf(Node*, const Tensor&){ /* no-op */ }

} // anon

// -------- dispatch table --------
VjpFn vjp_lookup(Op op){
    switch(op){
#define OP(name, arity, str) case Op::name: return &vjp_##name;
#include "ad/ops.def"
#undef OP
        default: return nullptr;
    }
}

} // namespace ag

// // ============================================
// // cgadimpl/src/autodiff/autodiff_jvp_ops.cpp
// // ============================================
// #include "ad/detail/autodiff_ops.hpp"

// namespace ag {
// namespace detail{

// // shorthand
// inline const Tensor& T(const std::function<const Tensor&(Node*)>& f, Node* p){ return f(p); }

// // ---- elementwise ----
// Tensor jvp_Add(Node* n, const std::function<const Tensor&(Node*)>& t){ 
//     return T(t,n->inputs[0].get()) + T(t,n->inputs[1].get());
// }
// Tensor jvp_Sub(Node* n, const std::function<const Tensor&(Node*)>& t){ 
//     return T(t,n->inputs[0].get()) - T(t,n->inputs[1].get());
// }
// Tensor jvp_Mul(Node* n, const std::function<const Tensor&(Node*)>& t){ 
//     Node* A=n->inputs[0].get(); Node* B=n->inputs[1].get();
//     return (T(t,A) * B->value) + (A->value * T(t,B));
// }
// Tensor jvp_Relu(Node* n, const std::function<const Tensor&(Node*)>& t){
//     Node* X=n->inputs[0].get();
//     int R=n->value.rows(), C=n->value.cols();
//     Tensor mask(R,C);
//     for(int i=0;i<R;++i) for(int j=0;j<C;++j) mask(i,j) = (n->value(i,j)>0.f)?1.f:0.f;
//     return T(t,X) * mask;
// }
// Tensor jvp_Exp(Node* n, const std::function<const Tensor&(Node*)>& t){
//     Node* X=n->inputs[0].get(); return T(t,X) * Tensor::exp(X->value);
// }
// Tensor jvp_Log(Node* n, const std::function<const Tensor&(Node*)>& t){
//     Node* X=n->inputs[0].get(); return T(t,X) / X->value;
// }
// Tensor jvp_Tanh(Node* n, const std::function<const Tensor&(Node*)>& t){
//     Node* X=n->inputs[0].get(); Tensor th=n->value, one=Tensor::ones_like(th);
//     return T(t,X) * (one - th*th);
// }
// Tensor jvp_Sigmoid(Node* n, const std::function<const Tensor&(Node*)>& t){
//     Node* X=n->inputs[0].get(); Tensor s=Tensor::sigmoid(X->value);
//     return T(t,X) * ( s * (Tensor::ones_like(s)-s) );
// }
// Tensor jvp_Softplus(Node* n, const std::function<const Tensor&(Node*)>& t){
//     Node* X=n->inputs[0].get(); return T(t,X) * Tensor::sigmoid(X->value);
// }
// Tensor jvp_SiLU(Node* n, const std::function<const Tensor&(Node*)>& t){
//     Node* X=n->inputs[0].get(); Tensor s=Tensor::sigmoid(X->value);
//     return T(t,X) * ( s + X->value * ( s * (Tensor::ones_like(s)-s) ) );
// }
// Tensor jvp_GELU(Node* n, const std::function<const Tensor&(Node*)>& t){
//     Node* X=n->inputs[0].get();
//     constexpr float c = 0.79788456080286535588f;
//     int R=X->value.rows(), C=X->value.cols();
//     Tensor x=X->value,u(R,C),dudx(R,C);
//     for(int i=0;i<R;++i) for(int j=0;j<C;++j){
//         float z=x(i,j);
//         u(i,j)=c*(z+0.044715f*z*z*z);
//         dudx(i,j)=c*(1.f+0.134145f*z*z);
//     }
//     Tensor th=Tensor::tanh(u), one=Tensor::ones_like(th);
//     Tensor dgelu=(one+th)*0.5f + (x * ((one - th*th) * dudx))*0.5f;
//     return T(t,X) * dgelu;
// }
// Tensor jvp_LeakyRelu(Node* n, const std::function<const Tensor&(Node*)>& t){
//     Node* X=n->inputs[0].get(); Node* A=n->inputs[1].get(); float a=A->value(0,0);
//     int R=X->value.rows(), C=X->value.cols(); Tensor out(R,C);
//     for(int i=0;i<R;++i) for(int j=0;j<C;++j){
//         float z=X->value(i,j);
//         out(i,j) = T(t,X)(i,j) * (z>0.f ? 1.f : a);
//     }
//     return out;
// }

// // ---- matmul ----
// Tensor jvp_MatMul(Node* n, const std::function<const Tensor&(Node*)>& t){
//     Node* A=n->inputs[0].get(); Node* B=n->inputs[1].get();
//     return Tensor::matmul(T(t,A), B->value) + Tensor::matmul(A->value, T(t,B));
// }
// Tensor jvp_FMA(Node* n, const std::function<const Tensor&(Node*)>& t){
//         Node* A=n->inputs[0].get(); Node* B=n->inputs[1].get(); 
//     return Tensor::matmul(T(t,A), Tensor::transpose( B->value)) + Tensor::matmul(A->value, T(t,B))+ T(t,n->inputs[2].get());
// }

// Tensor jvp_Linear(Node* n, const std::function<const Tensor&(Node*)>& t){
//         Node* A=n->inputs[0].get(); Node* B=n->inputs[1].get(); 
//     return Tensor::matmul(T(t,A), B->value) + Tensor::matmul(A->value, T(t,B))+ T(t,n->inputs[2].get());
// }

// Tensor jvp_Attention(Node* n, const std::function<const Tensor
//     &(Node*)>& t){
//     return Tensor();
// }

// Tensor jvp_AlibiAttention(Node* n, const std::function<const Tensor&(Node*)>& t){
//     return Tensor();
// }

// Tensor jvp_RELUAtt(Node* n, const std::function<const Tensor&(Node*)>& t){
//     return Tensor();
// }
// Tensor jvp_SigAtt(Node* n, const std::function<const Tensor&(Node*)>& t){
//     return Tensor();
// }


// Tensor jvp_Div(Node* n, const std::function<const Tensor&(Node*)>& t){
//     return Tensor();
// }
// Tensor jvp_Reciprocal(Node* n, const std::function<const Tensor&(Node*)>& t){
//     return Tensor();
// }
// Tensor jvp_Sign(Node* n, const std::function<const Tensor&(Node*)>& t){
//     return Tensor();
// }

// Tensor jvp_Sqrt(Node* n, const std::function<const Tensor&(Node*)>& t){
//     return Tensor();
// }
// Tensor jvp_Relumask(Node* n, const std::function<const Tensor&(Node*)>& t){
//     return Tensor();
// }
// Tensor jvp_Cosh(Node* n, const std::function<const Tensor&(Node*)>& t){
//     return Tensor();
// }
// Tensor jvp_Sinh(Node* n, const std::function<const Tensor&(Node*)>& t){
//     return Tensor();
// }


// Tensor jvp_Cos(Node* n, const std::function<const Tensor&(Node*)>& t){
//     return Tensor();
// }
// Tensor jvp_Sin(Node* n, const std::function<const Tensor&(Node*)>& t){
//     return Tensor();
// }

// Tensor jvp_MOE(Node* n, const std::function<const Tensor&(Node*)>& t){
//     Node* X=n->inputs[0].get(); return T(t,X) * (Tensor::cos(X->value)-(X->value*Tensor::sin(X->value)));
// }

// Tensor jvp_MSELoss(Node* n, const std::function<const Tensor&(Node*)>& t){
//     Node* Z = n->inputs[0].get();
//     Node* Y = n->inputs[1].get();
//     int B = Z->value.rows(), C = Z->value.cols();
//     Tensor diff = Z->value - Y->value;
//     Tensor gZ = diff * (2.0f / float(B * C));
//     Tensor gY = -diff * (2.0f / float(B * C));
//     float dotY = (gY * t(Y)).sum_scalar();
//         float dotZ = (gZ * t(Z)).sum_scalar();

//     Tensor out(1,1); out(0,0) = dotZ + dotY; return out;
// }

// Tensor jvp_MAELoss(Node* n, const std::function<const Tensor&(Node*)>& t){
//     Node* Z = n->inputs[0].get();
//     Node* Y = n->inputs[1].get();
//     int B = Z->value.rows(), C = Z->value.cols();
//     Tensor diff = Tensor::sign(Z->value - Y->value);
//     Tensor gZ = diff * (1.0f / float(B * C));
//     Tensor gY = -diff * (1.0f / float(B * C));
//     float dotY = (gY * t(Y)).sum_scalar();
//         float dotZ = (gZ * t(Z)).sum_scalar();

//     Tensor out(1,1); out(0,0) = dotZ + dotY; return out;
// }

// Tensor jvp_GCU(Node* n, const std::function<const Tensor&(Node*)>& t){
//     Node* X=n->inputs[0].get(); return T(t,X) *  (Tensor::cos(X->value)-(X->value*Tensor::sin(X->value)));
// }

// Tensor jvp_Parcon(Node* n, const std::function<const Tensor&(Node*)>& t){
//     Node* X=n->inputs[0].get(); return T(t,X) * ( 2 *Tensor::ones_like(X->value)- 2*X->value  );
// }

// Tensor jvp_LiSHT(Node* n, const std::function<const Tensor&(Node*)>& t){
//     Node* X=n->inputs[0].get(); return T(t,X) * ( Tensor::tanh(X->value)+ (Tensor::sech(X->value)*Tensor::sech(X->value)*X->value ) );
// }

// Tensor jvp_Transpose(Node* n, const std::function<const Tensor&(Node*)>& t){
//     Node* X=n->inputs[0].get(); return Tensor::transpose(T(t,X));
// }

// Tensor jvp_SWIGLU(Node* n, const std::function<const Tensor&(Node*)>& t){
//     return Tensor();
// }

// Tensor jvp_Mish(Node* n, const std::function<const Tensor&(Node*)>& t){
//     Node* X=n->inputs[0].get(); return T(t,X) *(Tensor::tanh( Tensor::softplus(X->value) )-(  (X->value*Tensor::sigmoid(X->value))  / (Tensor::cosh( Tensor::softplus(X->value)*Tensor::cosh( Tensor::softplus(X->value) ))    )            ));
// }
// Tensor jvp_Gaus(Node* n, const std::function<const Tensor&(Node*)>& t){
//      Node* X=n->inputs[0].get(); return T(t,X) * -2*X->value*Tensor::exp(-1*X->value*X->value);
// }

// Tensor jvp_LayerNorm(Node* n, const std::function<const Tensor&(Node*)>& t){
//     return Tensor();
// }

// Tensor jvp_RMSNorm(Node* n, const std::function<const Tensor&(Node*)>& t){
//     Node* X = n->inputs[0].get();
//     Tensor rms = *n->tape[0];
//     Tensor y   = *n->tape[1];   // normalized x

//     // upstream dot
//     Tensor dot = Tensor::row_sum(T(t,X) * y);  // [batch x 1]

//    return  (T(t,X) / rms) - (y * dot / (rms*X->value.cols()));
// }

// Tensor jvp_Dyntanh(Node* n, const std::function<const Tensor&(Node*)>& t){
//     return Tensor();
// }   

// Tensor jvp_RealLayerNorm(Node* n, const std::function<const Tensor&(Node*)>& t){
//     return Tensor();
// } 

// Tensor jvp_RealRMSNorm(Node* n, const std::function<const Tensor&(Node*)>& t){
//     return Tensor();
// }


// // ---- reductions ----
// Tensor jvp_Sum(Node* n, const std::function<const Tensor&(Node*)>& t){
//     Node* X=n->inputs[0].get(); Tensor s(1,1); s(0,0) = t(X).sum_scalar(); return s;
// }
// Tensor jvp_RowSum(Node* n, const std::function<const Tensor&(Node*)>& t){
//     Node* X=n->inputs[0].get(); return Tensor::row_sum( t(X) );
// }
// Tensor jvp_RowMax(Node* n, const std::function<const Tensor&(Node*)>& t){
//     Node* X=n->inputs[0].get();
//     int R=X->value.rows(), C=X->value.cols();
//     Tensor m = Tensor::row_max(X->value), M(R,C);
//     for(int i=0;i<R;++i) for(int j=0;j<C;++j) M(i,j)=(X->value(i,j)==m(i,0))?1.f:0.f;
//     return Tensor::row_sum( t(X) * M );
// }
// Tensor jvp_MeanAll(Node* n, const std::function<const Tensor&(Node*)>& t){
//     Node* X=n->inputs[0].get(); float s=1.f/float(X->value.rows()*X->value.cols());
//     Tensor out(1,1); out(0,0)= t(X).sum_scalar()*s; return out;
// }

// // ---- softmax / losses ----
// Tensor jvp_SoftmaxRow(Node* n, const std::function<const Tensor&(Node*)>& t){
//     Node* Z=n->inputs[0].get(); Tensor y=n->value; Tensor dot=Tensor::row_sum(y * t(Z));
//     return y * ( t(Z) - dot );
// }
// Tensor jvp_LogSumExpRow(Node* n, const std::function<const Tensor&(Node*)>& t){
//     Node* Z=n->inputs[0].get(); Tensor y=Tensor::softmax_row(Z->value);
//     return Tensor::row_sum( y * t(Z) );
// }
// Tensor jvp_CeWithLogits(Node* n, const std::function<const Tensor&(Node*)>& t){
//     Node* Z=n->inputs[0].get(); Node* Y=n->inputs[1].get(); int B=Z->value.rows();
//     Tensor sm = Tensor::softmax_row(Z->value);
//     Tensor gZ = (sm - Y->value) * (1.0f/float(B));
//     float dotZ = (gZ * t(Z)).sum_scalar();
//     Tensor lse = Tensor::logsumexp_row(Z->value);
//     Tensor lsm = Z->value - lse;
//     Tensor gY  = lsm * (-1.0f/float(B));
//     float dotY = (gY * t(Y)).sum_scalar();
//     Tensor out(1,1); out(0,0) = dotZ + dotY; return out;
// }

// Tensor jvp_KLDivergence(Node* n, const std::function<const Tensor&(Node*)>& t){
//     // leave it
//     return Tensor();
// }

// Tensor jvp_Leaf(Node*, const std::function<const Tensor&(Node*)>&){
//     return Tensor(); // unused
// }

// } // anon



// // -------- dispatch table --------
// JvpFn jvp_lookup(Op op){
//     switch(op){
// #define OP(name, arity, str) case Op::name: return &detail::jvp_##name;
// #include "ad/detail/ops.def"
// #undef OP
//         default: return nullptr;
//     }
// }

// } // namespace ag
// ====================================================================
// FILE: cgadimpl/src/autodiff/autodiff_jvp_ops.cpp (GPU-Aware Version)
// ====================================================================
#include "ad/detail/autodiff_ops.hpp"
#include <stdexcept> // Required for std::runtime_error

namespace ag {
namespace detail{

// shorthand
inline const Tensor& T(const std::function<const Tensor&(Node*)>& f, Node* p){ return f(p); }

// ---- elementwise ----
Tensor jvp_Add(Node* n, const std::function<const Tensor&(Node*)>& t){ 
    if (n->value.is_cpu()) {
        return T(t,n->inputs[0].get()) + T(t,n->inputs[1].get());
    } else {
        throw std::runtime_error("JVP for Add on CUDA not implemented yet!");
    }
}
Tensor jvp_Sub(Node* n, const std::function<const Tensor&(Node*)>& t){ 
    if (n->value.is_cpu()) {
        return T(t,n->inputs[0].get()) - T(t,n->inputs[1].get());
    } else {
        throw std::runtime_error("JVP for Sub on CUDA not implemented yet!");
    }
}
Tensor jvp_Mul(Node* n, const std::function<const Tensor&(Node*)>& t){ 
    if (n->value.is_cpu()) {
        Node* A=n->inputs[0].get(); Node* B=n->inputs[1].get();
        return (T(t,A) * B->value) + (A->value * T(t,B));
    } else {
        throw std::runtime_error("JVP for Mul on CUDA not implemented yet!");
    }
}
Tensor jvp_Relu(Node* n, const std::function<const Tensor&(Node*)>& t){
    if (n->value.is_cpu()) {
        Node* X=n->inputs[0].get();
        Tensor mask = Tensor::relu_mask(n->value);
        return T(t,X) * mask;
    } else {
        throw std::runtime_error("JVP for Relu on CUDA not implemented yet!");
    }
}
Tensor jvp_Exp(Node* n, const std::function<const Tensor&(Node*)>& t){
    if (n->value.is_cpu()) {
        Node* X=n->inputs[0].get(); return T(t,X) * n->value;
    } else {
        throw std::runtime_error("JVP for Exp on CUDA not implemented yet!");
    }
}
Tensor jvp_Log(Node* n, const std::function<const Tensor&(Node*)>& t){
    if (n->value.is_cpu()) {
        Node* X=n->inputs[0].get(); return T(t,X) / X->value;
    } else {
        throw std::runtime_error("JVP for Log on CUDA not implemented yet!");
    }
}
Tensor jvp_Tanh(Node* n, const std::function<const Tensor&(Node*)>& t){
    if (n->value.is_cpu()) {
        Node* X=n->inputs[0].get(); Tensor th=n->value;
        return T(t,X) * (Tensor::ones_like(th) - th*th);
    } else {
        throw std::runtime_error("JVP for Tanh on CUDA not implemented yet!");
    }
}
Tensor jvp_Sigmoid(Node* n, const std::function<const Tensor&(Node*)>& t){
    if (n->value.is_cpu()) {
        Node* X=n->inputs[0].get(); Tensor s=n->value;
        return T(t,X) * ( s * (Tensor::ones_like(s)-s) );
    } else {
        throw std::runtime_error("JVP for Sigmoid on CUDA not implemented yet!");
    }
}
Tensor jvp_Softplus(Node* n, const std::function<const Tensor&(Node*)>& t){
    if (n->value.is_cpu()) {
        Node* X=n->inputs[0].get(); return T(t,X) * Tensor::sigmoid(X->value);
    } else {
        throw std::runtime_error("JVP for Softplus on CUDA not implemented yet!");
    }
}
Tensor jvp_SiLU(Node* n, const std::function<const Tensor&(Node*)>& t){
    if (n->value.is_cpu()) {
        Node* X=n->inputs[0].get(); Tensor s=Tensor::sigmoid(X->value);
        return T(t,X) * ( s + X->value * ( s * (Tensor::ones_like(s)-s) ) );
    } else {
        throw std::runtime_error("JVP for SiLU on CUDA not implemented yet!");
    }
}
Tensor jvp_GELU(Node* n, const std::function<const Tensor&(Node*)>& t){
    if (n->value.is_cpu()) {
        Node* X=n->inputs[0].get();
        constexpr float c = 0.79788456080286535588f;
        Tensor x=X->value;
        Tensor u=c*(x+0.044715f*x*x*x);
        Tensor dudx=c*(1.f+0.134145f*x*x);
        Tensor th=Tensor::tanh(u);
        Tensor one=Tensor::ones_like(th);
        Tensor dgelu=(one+th)*0.5f + (x * ((one - th*th) * dudx))*0.5f;
        return T(t,X) * dgelu;
    } else {
        throw std::runtime_error("JVP for GELU on CUDA not implemented yet!");
    }
}
Tensor jvp_LeakyRelu(Node* n, const std::function<const Tensor&(Node*)>& t){
    if (n->value.is_cpu()) {
        Node* X=n->inputs[0].get(); Node* A=n->inputs[1].get(); float a=A->value(0,0);
        Tensor out = Tensor::zeros_like(X->value);
        for(int i=0;i<X->value.rows();++i) for(int j=0;j<X->value.cols();++j){
            out(i,j) = T(t,X)(i,j) * (X->value(i,j)>0.f ? 1.f : a);
        }
        return out;
    } else {
        throw std::runtime_error("JVP for LeakyRelu on CUDA not implemented yet!");
    }
}

// ---- matmul ----
Tensor jvp_MatMul(Node* n, const std::function<const Tensor&(Node*)>& t){
    if (n->value.is_cpu()) {
        Node* A=n->inputs[0].get(); Node* B=n->inputs[1].get();
        return Tensor::matmul(T(t,A), B->value) + Tensor::matmul(A->value, T(t,B));
    } else {
        throw std::runtime_error("JVP for MatMul on CUDA not implemented yet!");
    }
}
Tensor jvp_FMA(Node* n, const std::function<const Tensor&(Node*)>& t){
    if (n->value.is_cpu()) {
        Node* A=n->inputs[0].get(); Node* B=n->inputs[1].get(); Node* C=n->inputs[2].get();
        return Tensor::matmul(T(t,A), Tensor::transpose(B->value)) + Tensor::matmul(A->value, T(t,B)) + T(t,C);
    } else {
        throw std::runtime_error("JVP for FMA on CUDA not implemented yet!");
    }
}

Tensor jvp_Linear(Node* n, const std::function<const Tensor&(Node*)>& t){
    if (n->value.is_cpu()) {
        Node* A=n->inputs[0].get(); Node* B=n->inputs[1].get(); Node* C=n->inputs[2].get();
        return Tensor::matmul(T(t,A), B->value) + Tensor::matmul(A->value, T(t,B)) + T(t,C);
    } else {
        throw std::runtime_error("JVP for Linear on CUDA not implemented yet!");
    }
}

Tensor jvp_Attention(Node* n, const std::function<const Tensor&(Node*)>& t){
    throw std::runtime_error("JVP for Attention not implemented yet!");
}

Tensor jvp_AlibiAttention(Node* n, const std::function<const Tensor&(Node*)>& t){
    throw std::runtime_error("JVP for AlibiAttention not implemented yet!");
}

Tensor jvp_RELUAtt(Node* n, const std::function<const Tensor&(Node*)>& t){
    throw std::runtime_error("JVP for RELUAtt not implemented yet!");
}
Tensor jvp_SigAtt(Node* n, const std::function<const Tensor&(Node*)>& t){
    throw std::runtime_error("JVP for SigAtt not implemented yet!");
}

Tensor jvp_Div(Node* n, const std::function<const Tensor&(Node*)>& t){
    if (n->value.is_cpu()) {
        Node* A=n->inputs[0].get(); Node* B=n->inputs[1].get();
        Tensor recip_B = Tensor::reciprocal(B->value);
        return (T(t,A) * recip_B) - (A->value * recip_B * recip_B * T(t,B));
    } else {
        throw std::runtime_error("JVP for Div on CUDA not implemented yet!");
    }
}
Tensor jvp_Reciprocal(Node* n, const std::function<const Tensor&(Node*)>& t){
    if (n->value.is_cpu()) {
        Node* X=n->inputs[0].get();
        Tensor recip_X = Tensor::reciprocal(X->value);
        return -T(t,X) * recip_X * recip_X;
    } else {
        throw std::runtime_error("JVP for Reciprocal on CUDA not implemented yet!");
    }
}
Tensor jvp_Sign(Node* n, const std::function<const Tensor&(Node*)>& t){
    return Tensor::zeros_like(n->value); // Gradient is 0 almost everywhere
}

Tensor jvp_Sqrt(Node* n, const std::function<const Tensor&(Node*)>& t){
    if (n->value.is_cpu()) {
        return T(t,n->inputs[0].get()) * 0.5f * Tensor::reciprocal(n->value);
    } else {
        throw std::runtime_error("JVP for Sqrt on CUDA not implemented yet!");
    }
}
Tensor jvp_Relumask(Node* n, const std::function<const Tensor&(Node*)>& t){
    return Tensor::zeros_like(n->value); // Gradient is 0
}
Tensor jvp_Cosh(Node* n, const std::function<const Tensor&(Node*)>& t){
    if (n->value.is_cpu()) {
        return T(t,n->inputs[0].get()) * Tensor::sinh(n->inputs[0].get()->value);
    } else {
        throw std::runtime_error("JVP for Cosh on CUDA not implemented yet!");
    }
}
Tensor jvp_Sinh(Node* n, const std::function<const Tensor&(Node*)>& t){
    if (n->value.is_cpu()) {
        return T(t,n->inputs[0].get()) * Tensor::cosh(n->inputs[0].get()->value);
    } else {
        throw std::runtime_error("JVP for Sinh on CUDA not implemented yet!");
    }
}

Tensor jvp_Cos(Node* n, const std::function<const Tensor&(Node*)>& t){
    if (n->value.is_cpu()) {
        return -T(t,n->inputs[0].get()) * Tensor::sin(n->inputs[0].get()->value);
    } else {
        throw std::runtime_error("JVP for Cos on CUDA not implemented yet!");
    }
}
Tensor jvp_Sin(Node* n, const std::function<const Tensor&(Node*)>& t){
    if (n->value.is_cpu()) {
        return T(t,n->inputs[0].get()) * Tensor::cos(n->inputs[0].get()->value);
    } else {
        throw std::runtime_error("JVP for Sin on CUDA not implemented yet!");
    }
}

Tensor jvp_MOE(Node* n, const std::function<const Tensor&(Node*)>& t){
    throw std::runtime_error("JVP for MOE not implemented yet!");
}

Tensor jvp_MSELoss(Node* n, const std::function<const Tensor&(Node*)>& t){
    if (n->value.is_cpu()) {
        Node* Z = n->inputs[0].get(); Node* Y = n->inputs[1].get();
        int N = Z->value.numel();
        Tensor diff = Z->value - Y->value;
        Tensor gZ = diff * (2.0f / float(N));
        Tensor gY = -diff * (2.0f / float(N));
        float dot = (gZ * t(Z)).sum_scalar() + (gY * t(Y)).sum_scalar();
        return Tensor::floten(dot);
    } else {
        throw std::runtime_error("JVP for MSELoss on CUDA not implemented yet!");
    }
}

Tensor jvp_MAELoss(Node* n, const std::function<const Tensor&(Node*)>& t){
    if (n->value.is_cpu()) {
        Node* Z = n->inputs[0].get(); Node* Y = n->inputs[1].get();
        int N = Z->value.numel();
        Tensor diff = Tensor::sign(Z->value - Y->value);
        Tensor gZ = diff * (1.0f / float(N));
        Tensor gY = -diff * (1.0f / float(N));
        float dot = (gZ * t(Z)).sum_scalar() + (gY * t(Y)).sum_scalar();
        return Tensor::floten(dot);
    } else {
        throw std::runtime_error("JVP for MAELoss on CUDA not implemented yet!");
    }
}

Tensor jvp_GCU(Node* n, const std::function<const Tensor&(Node*)>& t){
    if (n->value.is_cpu()) {
        Node* X=n->inputs[0].get();
        return T(t,X) * (Tensor::cos(X->value) - X->value * Tensor::sin(X->value));
    } else {
        throw std::runtime_error("JVP for GCU on CUDA not implemented yet!");
    }
}

Tensor jvp_Parcon(Node* n, const std::function<const Tensor&(Node*)>& t){
    if (n->value.is_cpu()) {
        Node* X=n->inputs[0].get(); return T(t,X) * ( 2.0f * Tensor::ones_like(X->value) - 2.0f * X->value );
    } else {
        throw std::runtime_error("JVP for Parcon on CUDA not implemented yet!");
    }
}

Tensor jvp_LiSHT(Node* n, const std::function<const Tensor&(Node*)>& t){
    if (n->value.is_cpu()) {
        Node* X=n->inputs[0].get();
        Tensor sech_x = Tensor::sech(X->value);
        return T(t,X) * ( Tensor::tanh(X->value) + (sech_x*sech_x*X->value ) );
    } else {
        throw std::runtime_error("JVP for LiSHT on CUDA not implemented yet!");
    }
}

Tensor jvp_Transpose(Node* n, const std::function<const Tensor&(Node*)>& t){
    return Tensor::transpose(T(t,n->inputs[0].get()));
}

Tensor jvp_SWIGLU(Node* n, const std::function<const Tensor&(Node*)>& t){
    throw std::runtime_error("JVP for SWIGLU not implemented yet!");
}

Tensor jvp_Mish(Node* n, const std::function<const Tensor&(Node*)>& t){
    if (n->value.is_cpu()) {
        Node* X=n->inputs[0].get();
        Tensor sp = Tensor::softplus(X->value);
        Tensor th_sp = Tensor::tanh(sp);
        Tensor sig_x = Tensor::sigmoid(X->value);
        return T(t,X) * (th_sp + X->value * (Tensor::ones_like(th_sp) - th_sp*th_sp) * sig_x);
    } else {
        throw std::runtime_error("JVP for Mish on CUDA not implemented yet!");
    }
}
Tensor jvp_Gaus(Node* n, const std::function<const Tensor&(Node*)>& t){
    if (n->value.is_cpu()) {
        Node* X=n->inputs[0].get();
        return T(t,X) * -2.0f * X->value * n->value; // n->value is exp(-x^2)
    } else {
        throw std::runtime_error("JVP for Gaus on CUDA not implemented yet!");
    }
}

Tensor jvp_LayerNorm(Node* n, const std::function<const Tensor&(Node*)>& t){
    throw std::runtime_error("JVP for LayerNorm not implemented yet!");
}

Tensor jvp_RMSNorm(Node* n, const std::function<const Tensor&(Node*)>& t){
    if (n->value.is_cpu()) {
        Node* X = n->inputs[0].get();
        Tensor rms = *n->tape[0];
        Tensor y   = *n->tape[1];
        Tensor dot = Tensor::row_sum(T(t,X) * y);
        return (T(t,X) / rms) - (y * dot / (rms*X->value.cols()));
    } else {
        throw std::runtime_error("JVP for RMSNorm on CUDA not implemented yet!");
    }
}

Tensor jvp_Dyntanh(Node* n, const std::function<const Tensor&(Node*)>& t){
    throw std::runtime_error("JVP for Dyntanh not implemented yet!");
}   

Tensor jvp_RealLayerNorm(Node* n, const std::function<const Tensor&(Node*)>& t){
    throw std::runtime_error("JVP for RealLayerNorm not implemented yet!");
} 

Tensor jvp_RealRMSNorm(Node* n, const std::function<const Tensor&(Node*)>& t){
    throw std::runtime_error("JVP for RealRMSNorm not implemented yet!");
}

// ---- reductions ----
Tensor jvp_Sum(Node* n, const std::function<const Tensor&(Node*)>& t){
    if (n->value.is_cpu()) {
        Node* X=n->inputs[0].get();
        return Tensor::floten(t(X).sum_scalar());
    } else {
        throw std::runtime_error("JVP for Sum on CUDA not implemented yet!");
    }
}
Tensor jvp_RowSum(Node* n, const std::function<const Tensor&(Node*)>& t){
    if (n->value.is_cpu()) {
        Node* X=n->inputs[0].get(); return Tensor::row_sum( t(X) );
    } else {
        throw std::runtime_error("JVP for RowSum on CUDA not implemented yet!");
    }
}
Tensor jvp_RowMax(Node* n, const std::function<const Tensor&(Node*)>& t){
    if (n->value.is_cpu()) {
        Node* X=n->inputs[0].get();
        Tensor m = Tensor::row_max(X->value);
        Tensor M = Tensor::zeros_like(X->value);
        for(int i=0;i<X->value.rows();++i) for(int j=0;j<X->value.cols();++j)
            M(i,j)=(X->value(i,j)==m(i,0))?1.f:0.f;
        return Tensor::row_sum( t(X) * M );
    } else {
        throw std::runtime_error("JVP for RowMax on CUDA not implemented yet!");
    }
}
Tensor jvp_MeanAll(Node* n, const std::function<const Tensor&(Node*)>& t){
    if (n->value.is_cpu()) {
        Node* X=n->inputs[0].get(); float s=1.f/float(X->value.numel());
        return Tensor::floten(t(X).sum_scalar()*s);
    } else {
        throw std::runtime_error("JVP for MeanAll on CUDA not implemented yet!");
    }
}

// ---- softmax / losses ----
Tensor jvp_SoftmaxRow(Node* n, const std::function<const Tensor&(Node*)>& t){
    if (n->value.is_cpu()) {
        Node* Z=n->inputs[0].get(); Tensor y=n->value; Tensor dot=Tensor::row_sum(y * t(Z));
        return y * ( t(Z) - dot );
    } else {
        throw std::runtime_error("JVP for SoftmaxRow on CUDA not implemented yet!");
    }
}
Tensor jvp_LogSumExpRow(Node* n, const std::function<const Tensor&(Node*)>& t){
    if (n->value.is_cpu()) {
        Node* Z=n->inputs[0].get(); Tensor y=Tensor::softmax_row(Z->value);
        return Tensor::row_sum( y * t(Z) );
    } else {
        throw std::runtime_error("JVP for LogSumExpRow on CUDA not implemented yet!");
    }
}
Tensor jvp_CeWithLogits(Node* n, const std::function<const Tensor&(Node*)>& t){
    if (n->value.is_cpu()) {
        Node* Z=n->inputs[0].get(); Node* Y=n->inputs[1].get(); int B=Z->value.rows();
        Tensor sm = Tensor::softmax_row(Z->value);
        Tensor gZ = (sm - Y->value) * (1.0f/float(B));
        Tensor lse = Tensor::logsumexp_row(Z->value);
        Tensor lsm = Z->value - lse;
        Tensor gY  = lsm * (-1.0f/float(B));
        float dot = (gZ * t(Z)).sum_scalar() + (gY * t(Y)).sum_scalar();
        return Tensor::floten(dot);
    } else {
        throw std::runtime_error("JVP for CeWithLogits on CUDA not implemented yet!");
    }
}

Tensor jvp_KLDivergence(Node* n, const std::function<const Tensor&(Node*)>& t){
    throw std::runtime_error("JVP for KLDivergence not implemented yet!");
}

Tensor jvp_Leaf(Node*, const std::function<const Tensor&(Node*)>&){
    return Tensor(); // unused
}

} // namespace detail


// -------- dispatch table --------
JvpFn jvp_lookup(Op op){
    switch(op){
#define OP(name, arity, str) case Op::name: return &detail::jvp_##name;
#include "ad/detail/ops.def"
#undef OP
        default: return nullptr;
    }
}

} // namespace ag
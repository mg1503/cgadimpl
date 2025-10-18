// =============================================
// cgadimpl/src/nn/nn.cpp
// =============================================
#include "nn/nn.hpp"
#include <cmath>
#include <cassert>
#include "tensor.hpp"

namespace ag::nn {

// --- helpers ---
static inline Tensor broadcast_col(const Tensor& v, int R, int C){
    assert(v.cols()==1 && v.rows()==R);
    Tensor y(R,C);
    for (int i=0;i<R;++i){ float vi=v(i,0); for(int j=0;j<C;++j) y(i,j)=vi; }
    return y;
}

// --- elementwise ---
Tensor relu(const Tensor& x){
    Tensor y(x.rows(), x.cols());
    for(int i=0;i<x.rows();++i) for(int j=0;j<x.cols();++j)
        y(i,j) = x(i,j)>0.f ? x(i,j) : 0.f;
    return y;
}
Tensor leaky_relu(const Tensor& x, float a){
    Tensor y(x.rows(), x.cols());
    for(int i=0;i<x.rows();++i) for(int j=0;j<x.cols();++j){
        float v=x(i,j); y(i,j) = v>0.f ? v : a*v;
    }
    return y;
}
Tensor sigmoid(const Tensor& x){
    Tensor y(x.rows(), x.cols());
    for(int i=0;i<x.rows();++i) for(int j=0;j<x.cols();++j){
        float v=x(i,j); y(i,j)=1.f/(1.f+std::exp(-v));
    }
    return y;
}
Tensor tanh(const Tensor& x){
    Tensor y(x.rows(), x.cols());
    for(int i=0;i<x.rows();++i) for(int j=0;j<x.cols();++j)
        y(i,j)=std::tanh(x(i,j));
    return y;
}
Tensor softplus(const Tensor& x){
    Tensor y(x.rows(), x.cols());
    for(int i=0;i<x.rows();++i) for(int j=0;j<x.cols();++j){
        float v=x(i,j);
        y(i,j)=std::log1pf(std::exp(-std::abs(v))) + std::max(v,0.f); // stable softplus
    }
    return y;
}
Tensor silu(const Tensor& x){
    Tensor y(x.rows(), x.cols());
    for(int i=0;i<x.rows();++i) for(int j=0;j<x.cols();++j){
        float v=x(i,j); float s=1.f/(1.f+std::exp(-v)); y(i,j)=v*s;
    }
    return y;
}
Tensor gelu(const Tensor& x){
    Tensor y(x.rows(), x.cols());
    constexpr float c = 0.7978845608028654f; // sqrt(2/pi)
    for(int i=0;i<x.rows();++i) for(int j=0;j<x.cols();++j){
        float v=x(i,j);
        float u=c*(v+0.044715f*v*v*v);
        y(i,j)=0.5f*v*(1.f+std::tanh(u));
    }
    return y;
}

// --- rowwise reductions / softmax ---
// Tensor row_max(const Tensor& x){
//     Tensor m(x.rows(),1);
//     for(int i=0;i<x.rows();++i){
//         float mm=x(i,0);
//         for(int j=1;j<x.cols();++j) mm = mm>x(i,j)?mm:x(i,j);
//         m(i,0)=mm;
//     }
//     return m;
// }
// Tensor row_sum(const Tensor& x){
//     Tensor s(x.rows(),1);
//     for(int i=0;i<x.rows();++i){
//         float a=0.f; for(int j=0;j<x.cols();++j) a+=x(i,j);
//         s(i,0)=a;
//     }
//     return s;
// }
Tensor logsumexp_row(const Tensor& x){
    Tensor m = Tensor::row_max(x);                   // [B,1]
    Tensor xm = x - broadcast_col(m, x.rows(), x.cols());
    Tensor e(x.rows(), x.cols());
    for(int i=0;i<x.rows();++i) for(int j=0;j<x.cols();++j) e(i,j)=std::exp(xm(i,j));
    Tensor s = Tensor::row_sum(e);                   // [B,1]
    Tensor out(s.rows(), s.cols());
    for(int i=0;i<s.rows();++i) out(i,0)=std::log(s(i,0)) + m(i,0);
    return out;
}
Tensor softmax_row(const Tensor& x){
    Tensor m = Tensor::row_max(x);
    Tensor xm = x - broadcast_col(m, x.rows(), x.cols());
    Tensor e(x.rows(), x.cols());
    for(int i=0;i<x.rows();++i) for(int j=0;j<x.cols();++j) e(i,j)=std::exp(xm(i,j));
    Tensor s = Tensor::row_sum(e);
    Tensor sb = broadcast_col(s, x.rows(), x.cols());
    Tensor y(x.rows(), x.cols());
    for(int i=0;i<x.rows();++i) for(int j=0;j<x.cols();++j) y(i,j)=e(i,j)/sb(i,j);
    return y;
}

// --- loss ---
Tensor cross_entropy_with_logits(const Tensor& Z, const Tensor& Y){
    assert(Z.rows()==Y.rows() && Z.cols()==Y.cols());
    Tensor lse = logsumexp_row(Z);                 // [B,1]
    Tensor Zm  = Z - broadcast_col(lse, Z.rows(), Z.cols());
    // sum(Y * log_softmax(Z), axis=1)
    Tensor rs = Tensor::row_sum( Zm * Y );                 // [B,1]
    // mean over batch and negate
    float sum = 0.f; for(int i=0;i<rs.rows();++i) sum += rs(i,0);
    Tensor out(1,1); out(0,0) = -sum / float(Z.rows());
    return out;
}

Tensor scaled_dot_product_attention(const Tensor& Q,
                                    const Tensor& K,
                                    const Tensor& V,
                                    const Tensor* mask,
                                    float scale) {
    // Shapes:
    // Q:[B,D], K:[C,D] (note: we use K^T inside), V:[C,E], mask (optional):[B,C]
    assert(Q.cols() == K.cols());
    assert(K.rows() == V.rows());
    const int B = Q.rows();
    const int C = K.rows();
    const int D = Q.cols();
    (void)D;

    // S = Q @ K^T
    Tensor KT = Tensor::transpose(K);            // [D,C]
    Tensor S  = Tensor::matmul(Q, KT);           // [B,C]

    // scale (e.g., 1/sqrt(D))
    if (scale != 1.0f) {
        for (int i=0;i<S.rows();++i)
            for (int j=0;j<S.cols();++j)
                S(i,j) *= scale;
    }

    // mask fill: set masked positions to a large negative
    if (mask) {
        assert(mask->rows()==B && mask->cols()==C);
        for (int i=0;i<B;++i){
            for (int j=0;j<C;++j){
                if ((*mask)(i,j) == 0.0f) S(i,j) = -1e9f;
            }
        }
    }

    // P = softmax_row(S), O = P @ V
    Tensor P = softmax_row(S);                  // [B,C]
    Tensor O = Tensor::matmul(P, V);            // [B,E]
    return O;
}

} // namespace ag::nn

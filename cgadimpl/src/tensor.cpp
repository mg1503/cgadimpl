// =====================
// file: src/tensor.cpp (implementations)
// =====================
#include <random>
#include <algorithm>
#include <stdexcept>
#include <ostream>
#include <iomanip>
#include <cmath>
#include "ad/tensor.hpp"


namespace ag {


namespace {
inline std::pair<int,int> bshape(int r1,int c1,int r2,int c2){
int R = r1>r2? r1 : r2; int C = c1>c2? c1 : c2;
bool ok = ((r1==r2) || r1==1 || r2==1) && ((c1==c2) || c1==1 || c2==1);
if(!ok) throw std::runtime_error("broadcast: incompatible shapes");
return {R,C};
}
inline int pick(int i, int dim){ return dim==1 ? 0 : i; }
}


Tensor::Tensor() = default;
Tensor::Tensor(int rows, int cols) : r(rows), c(cols), d(static_cast<std::size_t>(rows)*cols, 0.f) {}


Tensor Tensor::zeros(int r, int c){ return Tensor(r,c); }

Tensor Tensor::ones (int r, int c){ Tensor t(r,c); std::fill(t.d.begin(), t.d.end(), 1.f); return t; }
// random tensor wich generates a tensor with dimensions r x c with values from N(0,1) using the given seed by normal distribution
Tensor Tensor::randn(int r, int c, unsigned seed){ Tensor t(r,c); std::mt19937 gen(seed); std::normal_distribution<float> N(0.f,1.f); for(auto &x: t.d) x = N(gen); return t; }
Tensor Tensor::zeros_like(const Tensor& x){ return zeros(x.r, x.c); }
Tensor Tensor::ones_like (const Tensor& x){ return ones (x.r, x.c); }


int Tensor::rows() const { return r; }
int Tensor::cols() const { return c; }
std::pair<int,int> Tensor::shape() const { return {r,c}; }
std::size_t Tensor::size() const { return d.size(); }


float& Tensor::operator()(int i, int j){ return d[static_cast<std::size_t>(i)*c + j]; }
const float& Tensor::operator()(int i, int j) const { return d[static_cast<std::size_t>(i)*c + j]; }


Tensor& Tensor::add_(const Tensor& g){
if(r!=g.r || c!=g.c) throw std::runtime_error("add_: shape mismatch");
for(std::size_t i=0;i<d.size();++i) d[i]+=g.d[i];
return *this;
}


float Tensor::sum_scalar() const { float s=0.f; for(float x: d) s+=x; return s; }
Tensor Tensor::sum_all(const Tensor& X){ Tensor y(1,1); y(0,0) = X.sum_scalar(); return y; }


Tensor operator+(const Tensor& a, const Tensor& b){
auto [R,C] = bshape(a.r,a.c,b.r,b.c);
Tensor y(R,C);
for(int i=0;i<R;++i){ int ia = pick(i,a.r), ib = pick(i,b.r);
for(int j=0;j<C;++j){ int ja = pick(j,a.c), jb = pick(j,b.c);
y(i,j) = a(ia,ja) + b(ib,jb);
}
}
return y; }
Tensor operator-(const Tensor& a, const Tensor& b){
auto [R,C] = bshape(a.r,a.c,b.r,b.c);
Tensor y(R,C);
for(int i=0;i<R;++i){ int ia = pick(i,a.r), ib = pick(i,b.r);
for(int j=0;j<C;++j){ int ja = pick(j,a.c), jb = pick(j,b.c);
y(i,j) = a(ia,ja) - b(ib,jb);
}
}
return y; }
Tensor operator*(const Tensor& a, const Tensor& b){
auto [R,C] = bshape(a.r,a.c,b.r,b.c);
Tensor y(R,C);
for(int i=0;i<R;++i){ int ia = pick(i,a.r), ib = pick(i,b.r);
for(int j=0;j<C;++j){ int ja = pick(j,a.c), jb = pick(j,b.c);
y(i,j) = a(ia,ja) * b(ib,jb);
}
}
return y; }
Tensor operator-(const Tensor& x){ Tensor y(x.r,x.c); for(std::size_t i=0;i<x.d.size();++i) y.d[i] = -x.d[i]; return y; }
Tensor operator*(const Tensor& a, float s){ Tensor y(a.r,a.c); for(std::size_t i=0;i<a.d.size();++i) y.d[i]=a.d[i]*s; return y; }
Tensor operator*(float s, const Tensor& a){ return a*s; }


Tensor Tensor::relu(const Tensor& x){ Tensor y(x.r,x.c); for(std::size_t i=0;i<x.d.size();++i) y.d[i] = x.d[i] > 0.f ? x.d[i] : 0.f; return y; }
Tensor Tensor::relu_mask(const Tensor& x){ Tensor m(x.r,x.c); for(std::size_t i=0;i<x.d.size();++i) m.d[i] = x.d[i] > 0.f ? 1.f : 0.f; return m; }
Tensor Tensor::abs(const Tensor& x){ Tensor m(x.r,x.c); for(std::size_t i=0;i<x.d.size();++i) m.d[i] = x.d[i] >= 0.f ? x.d[i] : -x.d[i]; return m; }
Tensor Tensor::sign(const Tensor& x){ Tensor m(x.r,x.c); for(std::size_t i=0;i<x.d.size();++i) m.d[i] = x.d[i] >= 0.f ? 1.f : -1.f; return m; }



Tensor Tensor::transpose(const Tensor& x){ Tensor y(x.c, x.r); for(int i=0;i<x.r;++i) for(int j=0;j<x.c;++j) y(j,i)=x(i,j); return y; }


Tensor Tensor::reduce_to(const Tensor& G, const Tensor& like){
if(G.r==like.r && G.c==like.c) return G; // nothing to do
Tensor out(like.r, like.c);
for(int i=0;i<G.r;++i){ int oi = (like.r==1?0:i);
for(int j=0;j<G.c;++j){ int oj = (like.c==1?0:j); out(oi,oj) += G(i,j); }
}
return out;
}


Tensor Tensor::matmul(const Tensor& A, const Tensor& B){ if(A.c!=B.r) throw std::runtime_error("matmul: inner dim mismatch"); Tensor Y(A.r, B.c);
for(int i=0;i<A.r;++i){ for(int k=0;k<A.c;++k){ float aik=A(i,k); for(int j=0;j<B.c;++j){ Y(i,j) += aik * B(k,j); } } }
return Y; }

Tensor Tensor::exp(const Tensor& x){ Tensor y(x.r,x.c); for(size_t i=0;i<x.d.size();++i) y.d[i]=std::exp(x.d[i]); return y; }
Tensor Tensor::log(const Tensor& x){ Tensor y(x.r,x.c); for(size_t i=0;i<x.d.size();++i) y.d[i]=std::log(x.d[i]); return y; }
Tensor Tensor::tanh(const Tensor& x){ Tensor y(x.r,x.c); for(size_t i=0;i<x.d.size();++i) y.d[i]=std::tanh(x.d[i]); return y; }
Tensor Tensor::sigmoid(const Tensor& x){ Tensor y(x.r,x.c); for(size_t i=0;i<x.d.size();++i){ float z=x.d[i]; y.d[i]=1.f/(1.f+std::exp(-z)); } return y; }
Tensor Tensor::softplus(const Tensor& x){ Tensor y(x.r,x.c); for(size_t i=0;i<x.d.size();++i){ float z=x.d[i]; y.d[i]=std::log1p(std::exp(-std::fabs(z))) + std::max(z,0.f); } return y; }
Tensor Tensor::gelu_tanh(const Tensor& x){ Tensor y(x.r,x.c); const float c = std::sqrt(2.f/M_PI); for(size_t i=0;i<x.d.size();++i){ float z=x.d[i]; float u = c*(z + 0.044715f*z*z*z); y.d[i] = 0.5f*z*(1.f+std::tanh(u)); } return y; }
Tensor Tensor::leaky_relu(const Tensor& x, float a){ Tensor y(x.r,x.c); for(size_t i=0;i<x.d.size();++i){ float z=x.d[i]; y.d[i] = z>0.f? z : a*z; } return y; }
Tensor Tensor::cos(const Tensor& x){ Tensor y(x.r,x.c); for(size_t i=0;i<x.d.size();++i) y.d[i]=std::cos(x.d[i]); return y; }
Tensor Tensor::sin(const Tensor& x){ Tensor y(x.r,x.c); for(size_t i=0;i<x.d.size();++i) y.d[i]=std::sin(x.d[i]); return y; }
Tensor Tensor::cosh(const Tensor& x){ Tensor y(x.r,x.c); for(size_t i=0;i<x.d.size();++i) y.d[i]=std::cosh(x.d[i]); return y; }


Tensor operator/(const Tensor& a, const Tensor& b){
auto [R,C] = bshape(a.r,a.c,b.r,b.c); Tensor y(R,C);
for(int i=0;i<R;++i){ int ia=pick(i,a.r), ib=pick(i,b.r); for(int j=0;j<C;++j){ int ja=pick(j,a.c), jb=pick(j,b.c); y(i,j) = a(ia,ja)/b(ib,jb); }}
return y;
}


Tensor Tensor::row_sum(const Tensor& X){ Tensor y(X.r,1); for(int i=0;i<X.r;++i){ float s=0.f; for(int j=0;j<X.c;++j) s+=X(i,j); y(i,0)=s; } return y; }
Tensor Tensor::row_max(const Tensor& X){ Tensor y(X.r,1); for(int i=0;i<X.r;++i){ float m=X(i,0); for(int j=1;j<X.c;++j) m=std::max(m,X(i,j)); y(i,0)=m; } return y; }


Tensor Tensor::softmax_row(const Tensor& Z){
Tensor M = row_max(Z); // [R,1]
Tensor e = exp(Z - M); // broadcast
Tensor s = row_sum(e); // [R,1]
return e / s; // broadcast divide
}

// used for cross-entropy with logits
Tensor Tensor::logsumexp_row(const Tensor& Z){
    Tensor M = row_max(Z);
    Tensor e = exp(Z - M);
    Tensor s = row_sum(e);
    Tensor lse = log(s) + M; // broadcast add
    return lse; // [R,1]
}

// mean of all elements
Tensor Tensor::mean_all(const Tensor& X){ Tensor y(1,1); y(0,0) = X.sum_scalar() / float(X.r * X.c); return y; }

// to print the tensor in a readable format
std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    os << "Tensor (" << t.rows() << "x" << t.cols() << "):\n";
    for (int i = 0; i < t.rows(); ++i) {
        for (int j = 0; j < t.cols(); ++j) {
            os << t(i, j);
            if (j + 1 < t.cols()) os << ' ';
        }
        if (i + 1 < t.rows()) os << '\n';
    }
    return os; // enable chaining
}

} // namespace ag
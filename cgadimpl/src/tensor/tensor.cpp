// ====================================================================
// FILE: cgadimpl/src/tensor/tensor.cpp (The Complete and Correct Version)
// ====================================================================
#include "tensor.hpp"
#include <random>
#include <algorithm>
#include <stdexcept>
#include <ostream>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error("CUDA Error in " __FILE__ ":" + std::to_string(__LINE__) + ": " + std::string(cudaGetErrorString(err))); \
        } \
    } while (0)

namespace ag {

// --- Custom Deleters for the shared_ptr ---
void cpu_deleter(float* p) { delete[] p; }
void cuda_deleter(float* p) { CUDA_CHECK(cudaFree(p)); }

// --- Private Helpers ---
namespace {
    inline std::pair<int,int> bshape(int r1,int c1,int r2,int c2){
        bool row_ok = (r1==r2) || (r1==1) || (r2==1);
        bool col_ok = (c1==c2) || (c1==1) || (c2==1);
        if (!row_ok || !col_ok) throw std::runtime_error("Incompatible broadcast shapes");
        return {std::max(r1,r2), std::max(c1,c2)};
    }
    inline int pick(int i, int dim){ return dim==1 ? 0 : i; }
}

// --- Constructors ---
Tensor::Tensor() : data_ptr_(nullptr), r_(0), c_(0), dev_(Device::CPU) {}

Tensor::Tensor(int rows, int cols, Device dev) : r_(rows), c_(cols), dev_(dev) {
    const size_t n = numel();
    if (n == 0) { data_ptr_ = nullptr; return; }
    if (dev == Device::CPU) {
        data_ptr_ = std::shared_ptr<float>(new float[n], cpu_deleter);
    } else {
        float* d_ptr;
        CUDA_CHECK(cudaMalloc(&d_ptr, n * sizeof(float)));
        data_ptr_ = std::shared_ptr<float>(d_ptr, cuda_deleter);
    }
}

// --- Device & Shape Info ---
int Tensor::rows() const { return r_; }
int Tensor::cols() const { return c_; }
std::pair<int,int> Tensor::shape() const { return {r_, c_}; }
std::size_t Tensor::numel() const { return static_cast<std::size_t>(r_) * c_; }
std::size_t Tensor::size() const { return numel(); }

// --- CPU-only element access ---
float& Tensor::operator()(int i, int j) {
    if (is_cuda()) throw std::runtime_error("Cannot use operator() on a CUDA tensor.");
    return data_ptr_.get()[static_cast<size_t>(i) * c_ + j];
}
const float& Tensor::operator()(int i, int j) const {
    if (is_cuda()) throw std::runtime_error("Cannot use operator() on a CUDA tensor.");
    return data_ptr_.get()[static_cast<size_t>(i) * c_ + j];
}

// --- The `.to()` method ---
Tensor Tensor::to(Device target_dev) const {
    if (dev_ == target_dev) return *this;
    Tensor new_tensor(r_, c_, target_dev);
    const size_t n_bytes = numel() * sizeof(float);
    if (dev_ == Device::CPU && target_dev == Device::CUDA) {
        CUDA_CHECK(cudaMemcpy(new_tensor.data(), this->data(), n_bytes, cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(new_tensor.data(), this->data(), n_bytes, cudaMemcpyDeviceToHost));
    }
    return new_tensor;
}

// --- Factories ---
Tensor Tensor::zeros(int r, int c, Device dev) {
    Tensor t(r, c, dev);
    if (t.numel() > 0) {
        if (dev == Device::CPU) std::fill(t.data(), t.data() + t.numel(), 0.0f);
        else CUDA_CHECK(cudaMemset(t.data(), 0, t.numel() * sizeof(float)));
    }
    return t;
}





int Tensor::rows() const { return r; }
int Tensor::cols() const { return c; }
std::pair<int,int> Tensor::shape() const { return {r,c}; }
std::size_t Tensor::size() const { return d.size(); }


float& Tensor::operator()(int i, int j){ return d[static_cast<std::size_t>(i)*c + j]; }
const float& Tensor::operator()(int i, int j) const { return d[static_cast<std::size_t>(i)*c + j]; }


// Tensor& Tensor::add_(const Tensor& g){
// if(r!=g.r || c!=g.c) throw std::runtime_error("add_: shape mismatch");
// for(std::size_t i=0;i<d.size();++i) d[i]+=g.d[i];
// return *this;
// }

Tensor& Tensor::add_(const Tensor& g) {
    // If self (this) tensor is uninitialized, allocate like g
    if (d.empty()) {
        r = g.r;
        c = g.c;
        d.resize(static_cast<std::size_t>(r) * c, 0.f);
    }

    if (r != g.r || c != g.c)
        throw std::runtime_error("add_: shape mismatch");

    for (std::size_t i = 0; i < d.size(); ++i)
        d[i] += g.d[i];
    return *this;
}

float Tensor::sum_scalar() const { float s=0.f; for(float x: d) s+=x; return s; }
Tensor Tensor::sum_all(const Tensor& X){ Tensor y(1,1); y(0,0) = X.sum_scalar(); return y; }


Tensor operator+(const Tensor& a, const Tensor& b) {
    REQUIRE_CPU(a, "operator+"); REQUIRE_CPU(b, "operator+");
    auto [R,C] = bshape(a.rows(),a.cols(),b.rows(),b.cols());
    Tensor y(R,C);
    for(int i=0;i<R;++i){
        int ia=pick(i,a.rows()), ib=pick(i,b.rows());
        for(int j=0;j<C;++j){
            int ja=pick(j,a.cols()), jb=pick(j,b.cols());
            y(i,j)=a(ia,ja)+b(ib,jb);
        }
    }
    return y;
}

Tensor operator-(const Tensor& a, const Tensor& b) {
    REQUIRE_CPU(a, "operator-"); REQUIRE_CPU(b, "operator-");
    auto [R,C] = bshape(a.rows(),a.cols(),b.rows(),b.cols());
    Tensor y(R,C);
    for(int i=0;i<R;++i){
        int ia=pick(i,a.rows()), ib=pick(i,b.rows());
        for(int j=0;j<C;++j){
            int ja=pick(j,a.cols()), jb=pick(j,b.cols());
            y(i,j)=a(ia,ja)-b(ib,jb);
        }
    }
    return y;
}

Tensor operator*(const Tensor& a, const Tensor& b) {
    REQUIRE_CPU(a, "operator*"); REQUIRE_CPU(b, "operator*");
    auto [R,C] = bshape(a.rows(),a.cols(),b.rows(),b.cols());
    Tensor y(R,C);
    for(int i=0;i<R;++i){
        int ia=pick(i,a.rows()), ib=pick(i,b.rows());
        for(int j=0;j<C;++j){
            int ja=pick(j,a.cols()), jb=pick(j,b.cols());
            y(i,j)=a(ia,ja)*b(ib,jb);
        }
    }
    return y;
}

Tensor operator-(const Tensor& x) {
    REQUIRE_CPU(x, "unary-");
    Tensor y(x.rows(), x.cols());
    for(size_t i=0; i < x.numel(); ++i) y.data()[i] = -x.data()[i];
    return y;
}

Tensor operator*(const Tensor& a, float s) {
    REQUIRE_CPU(a, "scalar*");
    Tensor y(a.rows(), a.cols());
    for(size_t i=0; i < a.numel(); ++i) y.data()[i] = a.data()[i] * s;
    return y;
}

Tensor operator*(float s, const Tensor& a){ return a*s; }

Tensor operator+(const Tensor& a, float s){
    REQUIRE_CPU(a, "scalar+");
    Tensor y(a.rows(), a.cols());
    for(size_t i=0; i < a.numel(); ++i) y.data()[i] = a.data()[i] + s;
    return y;
}

Tensor operator+(float s, const Tensor& a){ return a+s; }

Tensor Tensor::relu(const Tensor& x) {
    REQUIRE_CPU(x, "relu");
    Tensor y(x.rows(), x.cols());
    for(size_t i=0; i < x.numel(); ++i) y.data()[i] = x.data()[i] > 0.f ? x.data()[i] : 0.f;
    return y;
}

Tensor Tensor::relu_mask(const Tensor& x) {
    REQUIRE_CPU(x, "relu_mask");
    Tensor y(x.rows(), x.cols());
    for(size_t i=0; i < x.numel(); ++i) y.data()[i] = x.data()[i] > 0.f ? 1.f : 0.f;
    return y;
}

Tensor Tensor::transpose(const Tensor& x) {
    REQUIRE_CPU(x, "transpose");
    Tensor y(x.cols(), x.rows(), x.device());
    for(int i=0; i < x.rows(); ++i) {
        for(int j=0; j < x.cols(); ++j) {
            y(j, i) = x(i, j);
        }
    }
    return y;
}

Tensor Tensor::reciprocal(const Tensor &x) {
    REQUIRE_CPU(x, "reciprocal");
    Tensor y(x.rows(), x.cols());
    for(size_t i=0; i < x.numel(); ++i) y.data()[i] = 1.0f / x.data()[i];
    return y;
}

Tensor Tensor::abs (const Tensor& x) {
    REQUIRE_CPU(x, "abs");
    Tensor y(x.rows(), x.cols());
    for(size_t i=0; i < x.numel(); ++i) y.data()[i] = std::abs(x.data()[i]);
    return y;
}

Tensor Tensor::sign (const Tensor& x) {
    REQUIRE_CPU(x, "sign");
    Tensor y(x.rows(), x.cols());
    for(size_t i=0; i < x.numel(); ++i) y.data()[i] = (x.data()[i] > 0.f) ? 1.f : ((x.data()[i] < 0.f) ? -1.f : 0.f);
    return y;
}

Tensor Tensor::reduce_to(const Tensor& G, const Tensor& like){
if(G.r==like.r && G.c==like.c) return G; // nothing to do
Tensor out(like.r, like.c);
for(int i=0;i<G.r;++i){ int oi = (like.r==1?0:i);
    for(int j=0;j<G.c;++j){ int oj = (like.c==1?0:j); out(oi,oj) += G(i,j); }
}
return out;
}


Tensor Tensor::matmul(const Tensor& A, const Tensor& B){ if(A.c!=B.r) throw std::runtime_error("matmul: inner dim mismatch"); Tensor Y(A.r, B.c);
// ...existing code...
for(int i=0;i<A.r;++i){ for(int k=0;k<A.c;++k){ float aik=A(i,k); 
    for(int j=0;j<B.c;++j){ Y(i,j) += aik * B(k,j); } } }
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
Tensor Tensor::sech(const Tensor& x){ Tensor y(x.r,x.c); for(size_t i=0;i<x.d.size();++i) y.d[i]=1.f/std::cosh(x.d[i]); return y; }
Tensor Tensor::sqrt(const Tensor& x){ Tensor y(x.r,x.c); for(size_t i=0;i<x.d.size();++i) y.d[i]=std::sqrt(x.d[i]); return y; }


Tensor operator/(const Tensor& a, const Tensor& b){
auto [R,C] = bshape(a.r,a.c,b.r,b.c); Tensor y(R,C);
for(int i=0;i<R;++i){ int ia=pick(i,a.r), ib=pick(i,b.r); 
    for(int j=0;j<C;++j){ int ja=pick(j,a.c), jb=pick(j,b.c); y(i,j) = a(ia,ja)/b(ib,jb); }
}
return y;
}


Tensor Tensor::row_sum(const Tensor& X){ Tensor y(X.r,1); for(int i=0;i<X.r;++i){ float s=0.f; for(int j=0;j<X.c;++j) s+=X(i,j); y(i,0)=s; } return y; }
Tensor Tensor::row_max(const Tensor& X){ Tensor y(X.r,1); for(int i=0;i<X.r;++i){ float m=X(i,0); for(int j=1;j<X.c;++j) m=std::max(m,X(i,j)); y(i,0)=m; } return y; }


Tensor Tensor::softmax_row(const Tensor& Z) {
    REQUIRE_CPU(Z, "softmax_row");
    Tensor m = Tensor::row_max(Z);
    Tensor Z_shifted = Z - m;
    Tensor exp_Z = Tensor::exp(Z_shifted);
    Tensor sum_exp = Tensor::row_sum(exp_Z);
    return exp_Z / sum_exp;
}

Tensor Tensor::logsumexp_row(const Tensor& Z) {
    REQUIRE_CPU(Z, "logsumexp_row");
    Tensor m = Tensor::row_max(Z);
    Tensor Z_shifted = Z - m;
    Tensor exp_Z = Tensor::exp(Z_shifted);
    Tensor sum_exp = Tensor::row_sum(exp_Z);
    return Tensor::log(sum_exp) + m;
}

// mean of all elements
Tensor Tensor::mean_all(const Tensor& X){ 
    Tensor y(1,1); y(0,0) = X.sum_scalar() / float(X.r * X.c); 
    return y; 
}

// Matmul is the one function we update fully
Tensor Tensor::matmul(const Tensor &A, const Tensor &B){
    if(A.cols() != B.rows()) throw std::runtime_error("matmul: inner dim mismatch");
    if(A.device() != B.device()) throw std::runtime_error("matmul: device mismatch");
    
    Tensor Y = Tensor::zeros(A.rows(), B.cols(), A.device());

    if (A.is_cpu()) {
        for(int i=0;i<A.rows();++i){
            for(int k=0;k<A.cols();++k){
                float aik=A(i,k);
                for(int j=0;j<B.cols();++j){
                    Y(i,j) += aik * B(k,j);
                }
            }
        }
    } else {
        throw std::runtime_error("Tensor::matmul for CUDA should be called via nodeops dispatch, not directly.");
    }
    return Y;
}

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    if (t.is_cuda()) {
        os << "Tensor(" << t.rows() << "x" << t.cols() << ", device=CUDA)";
    } else {
        os << "Tensor (" << t.rows() << "x" << t.cols() << ", device=CPU):\n";
        for (int i = 0; i < std::min(t.rows(), 10); ++i) {
            for (int j = 0; j < std::min(t.cols(), 10); ++j) {
                os << std::setw(10) << std::setprecision(4) << t(i, j) << " ";
            }
            if (t.cols() > 10) os << "...";
            os << "\n";
        }
        if (t.rows() > 10) os << "...\n";
    }
    return os;
}

} // namespace ag
// ====================================================================
// FILE: cgadimpl/src/tensor/tensor.cpp (The New Device-Aware Version)
// ====================================================================
#include "tensor.hpp"
#include <random>
#include <algorithm>
#include <stdexcept>
#include <ostream>
#include <iomanip>
#include <cmath>
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

Tensor Tensor::ones(int r, int c, Device dev) {
    Tensor t(r, c, dev);
    if (t.numel() > 0) {
        if (dev == Device::CPU) {
            std::fill(t.data(), t.data() + t.numel(), 1.0f);
        } else {
            std::vector<float> temp(t.numel(), 1.0f);
            CUDA_CHECK(cudaMemcpy(t.data(), temp.data(), t.numel() * sizeof(float), cudaMemcpyHostToDevice));
        }
    }
    return t;
}

Tensor Tensor::randn(int r, int c, unsigned seed, Device dev) {
    Tensor t_cpu(r, c, Device::CPU);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.f, 1.f);
    for (size_t i = 0; i < t_cpu.numel(); ++i) t_cpu.data()[i] = dist(gen);
    return dev == Device::CPU ? t_cpu : t_cpu.to(Device::CUDA);
}

Tensor Tensor::zeros_like(const Tensor& x) { return zeros(x.r_, x.c_, x.dev_); }
Tensor Tensor::ones_like(const Tensor& x) { return ones(x.r_, x.c_, x.dev_); }
Tensor Tensor::floten(float q) { Tensor t(1, 1); t(0,0) = q; return t; }
Tensor Tensor::alibi(int rows, int cols, float m) { /* Unchanged CPU-only code */ return Tensor(); }

// --- Grad accumulation ---
Tensor& Tensor::add_(const Tensor& g) {
    if (this->shape() != g.shape() || this->device() != g.device()) throw std::runtime_error("add_: shape or device mismatch");
    if (is_cpu()) {
        for(size_t i=0; i<numel(); ++i) data()[i] += g.data()[i];
    } else {
        // Here we would call an 'add' kernel. For now, we error.
        throw std::runtime_error("add_ for CUDA not implemented yet");
    }
    return *this;
}

// --- ALL OTHER FUNCTIONS ARE NOW CPU-ONLY (will throw error if called on GPU tensor) ---
#define REQUIRE_CPU(tensor, func_name) \
    if ((tensor).is_cuda()) throw std::runtime_error(std::string(func_name) + " is CPU-only for now.")

float Tensor::sum_scalar() const { REQUIRE_CPU(*this, "sum_scalar"); float s=0.f; for(size_t i=0; i<numel(); ++i) s += data()[i]; return s; }
Tensor Tensor::sum_all(const Tensor& X) { REQUIRE_CPU(X, "sum_all"); Tensor y(1,1); y(0,0) = X.sum_scalar(); return y; }
Tensor operator+(const Tensor& a, const Tensor& b) { REQUIRE_CPU(a, "operator+"); REQUIRE_CPU(b, "operator+"); auto [R,C] = bshape(a.rows(),a.cols(),b.rows(),b.cols()); Tensor y(R,C); for(int i=0;i<R;++i){ int ia=pick(i,a.rows()),ib=pick(i,b.rows()); for(int j=0;j<C;++j){ int ja=pick(j,a.cols()),jb=pick(j,b.cols()); y(i,j)=a(ia,ja)+b(ib,jb);}} return y;}
Tensor operator-(const Tensor& a, const Tensor& b) { REQUIRE_CPU(a, "operator-"); REQUIRE_CPU(b, "operator-"); /* ... old code ... */ return Tensor();}
Tensor operator*(const Tensor& a, const Tensor& b) { REQUIRE_CPU(a, "operator*"); REQUIRE_CPU(b, "operator*"); /* ... old code ... */ return Tensor();}
Tensor operator-(const Tensor& x) { REQUIRE_CPU(x, "unary-"); /* ... old code ... */ return Tensor();}
Tensor operator*(const Tensor& a, float s) { REQUIRE_CPU(a, "scalar*"); /* ... old code ... */ return Tensor();}
Tensor operator*(float s, const Tensor& a){ return a*s; }
Tensor operator+(const Tensor& a, float s){ REQUIRE_CPU(a, "scalar+"); /* ... old code ... */ return Tensor();}
Tensor operator+(float s, const Tensor& a){ return a+s; }
Tensor Tensor::relu (const Tensor& x) { REQUIRE_CPU(x, "relu"); /* ... old code ... */ return Tensor();}
Tensor Tensor::relu_mask(const Tensor& x) { REQUIRE_CPU(x, "relu_mask"); /* ... old code ... */ return Tensor();}
Tensor Tensor::transpose(const Tensor& x) { REQUIRE_CPU(x, "transpose"); /* ... old code ... */ return Tensor();}
Tensor Tensor::reciprocal(const Tensor &x) { REQUIRE_CPU(x, "reciprocal"); /* ... old code ... */ return Tensor();}
Tensor Tensor::abs (const Tensor& x) { REQUIRE_CPU(x, "abs"); /* ... old code ... */ return Tensor();}
Tensor Tensor::sign (const Tensor& x) { REQUIRE_CPU(x, "sign"); /* ... old code ... */ return Tensor();}
Tensor Tensor::reduce_to(const Tensor& G, const Tensor& like) { REQUIRE_CPU(G, "reduce_to"); /* ... old code ... */ return Tensor();}
Tensor Tensor::sinh(const Tensor &x) { REQUIRE_CPU(x, "sinh"); /* ... old code ... */ return Tensor();}
Tensor Tensor::exp(const Tensor& x) { REQUIRE_CPU(x, "exp"); /* ... old code ... */ return Tensor();}
Tensor Tensor::log(const Tensor& x) { REQUIRE_CPU(x, "log"); /* ... old code ... */ return Tensor();}
// ... etc. for all other math functions ...
Tensor Tensor::cos(const Tensor& x) { REQUIRE_CPU(x, "cos"); return Tensor(); }
Tensor Tensor::sin(const Tensor& x) { REQUIRE_CPU(x, "sin"); return Tensor(); }
Tensor Tensor::cosh(const Tensor& x) { REQUIRE_CPU(x, "cosh"); return Tensor(); }
Tensor Tensor::sech(const Tensor& x) { REQUIRE_CPU(x, "sech"); return Tensor(); }
Tensor Tensor::sqrt(const Tensor &x) { REQUIRE_CPU(x, "sqrt"); return Tensor(); }
Tensor Tensor::tanh(const Tensor& x) { REQUIRE_CPU(x, "tanh"); return Tensor(); }
Tensor Tensor::sigmoid(const Tensor& x) { REQUIRE_CPU(x, "sigmoid"); return Tensor(); }
Tensor Tensor::softplus(const Tensor& x) { REQUIRE_CPU(x, "softplus"); return Tensor(); }
Tensor Tensor::gelu_tanh(const Tensor& x) { REQUIRE_CPU(x, "gelu_tanh"); return Tensor(); }
Tensor Tensor::leaky_relu(const Tensor& x, float alpha) { REQUIRE_CPU(x, "leaky_relu"); return Tensor(); }
Tensor operator/(const Tensor& a, const Tensor& b) { REQUIRE_CPU(a, "operator/"); REQUIRE_CPU(b, "operator/"); return Tensor(); }
Tensor Tensor::row_sum(const Tensor& X) { REQUIRE_CPU(X, "row_sum"); return Tensor(); }
Tensor Tensor::row_max(const Tensor& X) { REQUIRE_CPU(X, "row_max"); return Tensor(); }
Tensor Tensor::softmax_row(const Tensor& Z) { REQUIRE_CPU(Z, "softmax_row"); return Tensor(); }
Tensor Tensor::logsumexp_row(const Tensor& Z) { REQUIRE_CPU(Z, "logsumexp_row"); return Tensor(); }
Tensor Tensor::mean_all(const Tensor& X) { REQUIRE_CPU(X, "mean_all"); return Tensor(); }

// Matmul is the one function we update fully
Tensor Tensor::matmul(const Tensor &A, const Tensor &B){
    if(A.cols() != B.rows()) throw std::runtime_error("matmul: inner dim mismatch");
    if(A.device() != B.device()) throw std::runtime_error("matmul: device mismatch");
    
    Tensor Y(A.rows(), B.cols(), A.device());

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
        // This is where we will wire up the kernel in the nodeops file.
        // For now, this static function will throw.
        throw std::runtime_error("Tensor::matmul for CUDA should be called via nodeops dispatch, not directly.");
    }
    return Y;
}

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    if (t.is_cuda()) {
        os << "Tensor(" << t.rows() << "x" << t.cols() << ", device=CUDA)";
    } else {
        os << "Tensor (" << t.rows() << "x" << t.cols() << ", device=CPU):\n";
        for (int i = 0; i < std::min(t.rows(), 10); ++i) { // Print max 10 rows
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
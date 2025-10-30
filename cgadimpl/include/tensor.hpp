// // ====================================================================
// // FILE: cgadimpl/include/ad/tensor.hpp (The New Device-Aware Version)
// // ====================================================================
// #pragma once
// #include <cstddef>
// #include <utility>
// #include <vector>
// #include <iosfwd>
// #include <memory>

// namespace ag {

// // The new Device enum, replacing the old 'bool on_cuda'
// enum class Device { CPU, CUDA };

// class Tensor {
// private:
//     // CRITICAL CHANGE: The storage is now a shared_ptr.
//     std::shared_ptr<float> data_ptr_;
//     int r_{0}, c_{0};
//     Device dev_{Device::CPU};

// public:
//     // --- Constructors & Destructor ---
//     Tensor();
//     Tensor(int rows, int cols, Device dev = Device::CPU);

//     // --- Device Info & Control ---
//     Device device() const noexcept { return dev_; }
//     bool is_cpu()   const noexcept { return dev_ == Device::CPU; }
//     bool is_cuda()  const noexcept { return dev_ == Device::CUDA; }
//     Tensor to(Device target_dev) const;

//     // --- Inline helper from your original file ---
//     inline Tensor rt(const Tensor& g, const Tensor& like){ return Tensor::reduce_to(g, like); }

//     // --- Factories (now take a Device enum) ---
//     static Tensor zeros(int r, int c, Device dev = Device::CPU);
//     static Tensor ones (int r, int c, Device dev = Device::CPU);
//     static Tensor randn(int r, int c, unsigned seed=42, Device dev = Device::CPU);
//     static Tensor zeros_like(const Tensor& x);
//     static Tensor ones_like (const Tensor& x);

//     // --- Data Access ---
//     float* data() { return data_ptr_.get(); }
//     const float* data() const { return data_ptr_.get(); }

//     // --- Shape/Info ---
//     int rows() const;
//     int cols() const;
//     std::pair<int,int> shape() const;
//     std::size_t numel() const;
//     std::size_t size() const;

//     // --- CPU-only element access (throws error if on CUDA) ---
//     float& operator()(int i, int j);
//     const float& operator()(int i, int j) const;

//     // --- Grad accumulation ---
//     Tensor& add_(const Tensor& g);

//     // --- ALL ORIGINAL FUNCTIONS ARE PRESERVED ---
//     float sum_scalar() const;
//     static Tensor sum_all(const Tensor& X);
//     friend Tensor operator+(const Tensor& a, const Tensor& b);
//     friend Tensor operator-(const Tensor& a, const Tensor& b);
//     friend Tensor operator*(const Tensor& a, const Tensor& b);
//     friend Tensor operator-(const Tensor& x);
//     friend Tensor operator*(const Tensor& a, float s);
//     friend Tensor operator*(float s, const Tensor& a);
//     friend Tensor operator+(const Tensor& a, float s);
//     friend Tensor operator+(float s, const Tensor& a);
//     static Tensor relu (const Tensor& x);
//     static Tensor relu_mask(const Tensor& x);
//     static Tensor transpose(const Tensor& x);
//     static Tensor reciprocal(const Tensor &x);
//     static Tensor matmul(const Tensor &A, const Tensor &B);
//     static Tensor abs (const Tensor& x);
//     static Tensor sign (const Tensor& x);
//     static Tensor reduce_to(const Tensor& G, const Tensor& like);
//     static Tensor floten(float q);
//     static Tensor alibi(int rows, int cols, float m);
//     static Tensor sinh(const Tensor &x);
//     static Tensor exp(const Tensor& x);
//     static Tensor log(const Tensor& x);
//     static Tensor cos(const Tensor& x);
//     static Tensor sin(const Tensor& x);
//     static Tensor cosh(const Tensor& x);
//     static Tensor sech(const Tensor& x);
//     static Tensor sqrt(const Tensor &x);
//     static Tensor tanh(const Tensor& x);
//     static Tensor sigmoid(const Tensor& x);
//     static Tensor softplus(const Tensor& x);
//     static Tensor gelu_tanh(const Tensor& x);
//     static Tensor leaky_relu(const Tensor& x, float alpha);
//     friend Tensor operator/(const Tensor& a, const Tensor& b);
//     static Tensor row_sum(const Tensor& X);
//     static Tensor row_max(const Tensor& X);
//     static Tensor softmax_row(const Tensor& Z);
//     static Tensor logsumexp_row(const Tensor& Z);
//     static Tensor mean_all(const Tensor& X);
//     friend std::ostream& operator<<(std::ostream& os, const Tensor& t);

//     static Tensor silu(const Tensor& x);
//     static Tensor gelu(const Tensor& x);
// };

// } // namespace ag


#pragma once
#include "TensorLib.h"


namespace ag {
    using namespace OwnTensor;

    inline TensorOptions options(const Tensor& t) {
        return TensorOptions().with_dtype(t.dtype()).with_device(t.device()).with_req_grad(t.requires_grad());
    }
} // namespace ag       
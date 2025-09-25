// =====================
// file: include/ag/tensor.hpp (declarations only)
// =====================
#pragma once
#include <cstddef>
#include <utility>
#include <vector>
#include <iosfwd> // for std::ostream forward decl


namespace ag {


class Tensor {
public:
// ctors
Tensor();
Tensor(int rows, int cols);


// factories
static Tensor zeros(int r, int c);
static Tensor ones (int r, int c);
static Tensor randn(int r, int c, unsigned seed=42);
static Tensor zeros_like(const Tensor& x);
static Tensor ones_like (const Tensor& x);


// shape/info
int rows() const;
int cols() const;
std::pair<int,int> shape() const;
std::size_t size() const;


// element access
float& operator()(int i, int j);
const float& operator()(int i, int j) const;


// grad accumulation utility
Tensor& add_(const Tensor& g);


// reductions
float sum_scalar() const;
static Tensor sum_all(const Tensor& X);


// Broadcasting-aware elementwise ops (NumPy-style for 2D):
// result shape: (max(r), max(c)); dimensions must match or be 1.
friend Tensor operator+(const Tensor& a, const Tensor& b);
friend Tensor operator-(const Tensor& a, const Tensor& b);
friend Tensor operator*(const Tensor& a, const Tensor& b); // Hadamard

// unary / scalar ops
friend Tensor operator-(const Tensor& x); // unary negation
friend Tensor operator*(const Tensor& a, float s); // scalar scale
friend Tensor operator*(float s, const Tensor& a); // scalar scale


static Tensor relu (const Tensor& x);
static Tensor relu_mask(const Tensor& x); // 1 where x>0 else 0
static Tensor transpose(const Tensor& x);
static Tensor matmul (const Tensor& A, const Tensor& B);

// Reduce G to the shape of `like` by summing broadcasted axes.
static Tensor reduce_to(const Tensor& G, const Tensor& like);


// elementwise unary
static Tensor exp(const Tensor& x);
static Tensor log(const Tensor& x);
static Tensor tanh(const Tensor& x);
static Tensor sigmoid(const Tensor& x);
static Tensor softplus(const Tensor& x);
static Tensor gelu_tanh(const Tensor& x); // tanh approx
static Tensor leaky_relu(const Tensor& x, float alpha);


// binary elementwise division (broadcasting)
friend Tensor operator/(const Tensor& a, const Tensor& b);


// rowwise reductions (produce [R,1])
static Tensor row_sum(const Tensor& X);
static Tensor row_max(const Tensor& X);


// softmax family (rowwise)
static Tensor softmax_row(const Tensor& Z);
static Tensor logsumexp_row(const Tensor& Z);


// averages
static Tensor mean_all(const Tensor& X);

// debug print
friend std::ostream& operator<<(std::ostream& os, const Tensor& t);


private:
int r{0}, c{0};
std::vector<float> d; // private storage
};


} // namespace ag
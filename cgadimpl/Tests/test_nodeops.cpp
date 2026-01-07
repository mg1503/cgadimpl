#include "ad/ag_all.hpp"
#include <cassert>
#include <vector> // Required for shape comparison

using namespace ag;
using namespace OwnTensor; // For Shape, TensorOptions

int main() {
  Tensor A = Tensor::randn<float>(Shape{{4,3}}, TensorOptions().with_req_grad(true));
  Tensor B = Tensor::randn<float>(Shape{{3,2}}, TensorOptions().with_req_grad(true));
  
  // --- FIX: Use the correct 3-argument Node constructor ---
  auto na = std::make_shared<Node>(A, Op::Leaf, "A");
  auto nb = std::make_shared<Node>(B, Op::Leaf, "B");

  auto nmm = ag::detail::matmul_nodeops(na, nb);

  // --- FIX: Use the modern, N-dimensional shape access ---
  auto expected_shape = std::vector<int64_t>{4, 2};
  assert(nmm->value.shape().dims == expected_shape);

  std::cout << "[OK] test_nodeops passed." << std::endl;
  
  return 0;
}
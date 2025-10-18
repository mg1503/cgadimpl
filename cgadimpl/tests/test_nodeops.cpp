// tests/test_nodeops.cpp
#include "ad/ag_all.hpp"
#include <cassert>
using namespace ag;

int main() {
  Tensor A = Tensor::randn(4,3);
  Tensor B = Tensor::randn(3,2);
  auto na = std::make_shared<Node>(A, true, Op::Leaf, "A");
  auto nb = std::make_shared<Node>(B, true, Op::Leaf, "B");

  auto nmm = ag::detail::matmul_nodeops(na, nb);
  assert(nmm->value.rows() == 4 && nmm->value.cols() == 2);

  return 0;
}

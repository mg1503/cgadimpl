// =====================
// file: cgadimpl/include/ag/optim.hpp (declarations only)
// =====================
#pragma once
#include "ad/ops/ops.hpp"
#include "ad/utils/debug.hpp"
#include "ad/autodiff/autodiff.hpp"
#include "ad/detail/autodiff_ops.hpp"
#include "tensor.hpp"
#include "ad/utils/debug.hpp"
#include <math.h>

namespace ag {

void SGD(const Value& root, const Tensor* grad_seed=nullptr, float learning_rate=100);

}
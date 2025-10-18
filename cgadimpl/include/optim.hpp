// =====================
// file: cgadimpl/include/ag/optim.hpp (declarations only)
// =====================
#pragma once
#include "ad/ops.hpp"
#include "ad/debug.hpp"
#include "ad/autodiff.hpp"
#include "ad/detail/autodiff_ops.hpp"
#include "tensor.hpp"
#include "ad/debug.hpp"
#include <math.h>

namespace ag {

void SGD(const Value& root, const Tensor* grad_seed=nullptr, int learning_rate=100);

}
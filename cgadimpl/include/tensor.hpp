// // ====================================================================
// // FILE: cgadimpl/include/ad/tensor.hpp (The New Device-Aware Version)
// // ====================================================================

#pragma once
#include "TensorLib.h"


namespace ag {
    using namespace OwnTensor;

    inline TensorOptions options(const Tensor& t) {
        return TensorOptions().with_dtype(t.dtype()).with_device(t.device()).with_req_grad(t.requires_grad());
    }
} // namespace ag       
#include "ad/optimizer/loss_scaler.hpp"
#include "ad/autodiff/autodiff.hpp"
#include "ad/detail/autodiff_ops.hpp"
#include <cmath>
#include <algorithm>

namespace ag {

LossScaler::LossScaler(float init_scale, int backoff_factor, int growth_factor, int growth_interval)
    : current_scale_(init_scale), backoff_factor_(backoff_factor), growth_factor_(growth_factor), 
      growth_interval_(growth_interval), steps_since_last_overflow_(0) {}

Value LossScaler::scale_loss(Value loss) {
    return loss * current_scale_;
}

bool LossScaler::unscale_gradients(const std::vector<Value>& params) {
    bool overflow = false;
    float inv_scale = 1.0f / current_scale_;

    for (const auto& p : params) {
        Node* n = p.node.get();
        if (!n->requires_grad()) continue;

        if (has_overflow(n->grad)) {
            overflow = true;
            break;
        }
    }

    if (!overflow) {
        for (const auto& p : params) {
            Node* n = p.node.get();
            if (!n->requires_grad()) continue;
            n->grad *= inv_scale;
        }
    }

    return overflow;
}

void LossScaler::update(bool overflow) {
    if (overflow) {
        current_scale_ /= backoff_factor_;
        if (current_scale_ < 1.0f) current_scale_ = 1.0f;
        steps_since_last_overflow_ = 0;
    } else {
        steps_since_last_overflow_++;
        if (steps_since_last_overflow_ >= growth_interval_) {
            current_scale_ *= growth_factor_;
            steps_since_last_overflow_ = 0;
        }
    }
}

bool LossScaler::has_overflow(const Tensor& grad) {
    // Check for inf/nan in the gradient
    // This is a simplified check for CPU. 
    // In a real implementation, we might want a specialized kernel.
    
    // For now, we'll use a simple loop on CPU.
    // We need to dispatch by dtype to access data.
    
    bool overflow = false;
    
    dispatch_by_dtype(grad.dtype(), [&](auto dummy) {
        using T = decltype(dummy);
        const T* data = grad.data<T>();
        size_t n = grad.numel();
        
        for (size_t i = 0; i < n; ++i) {
            if constexpr (std::is_same_v<T, complex32_t> || std::is_same_v<T, complex64_t> || std::is_same_v<T, complex128_t>) {
                float r = static_cast<float>(data[i].real());
                float im = static_cast<float>(data[i].imag());
                if (std::isinf(r) || std::isnan(r) || std::isinf(im) || std::isnan(im)) {
                    overflow = true;
                    break;
                }
            } else {
                float val = static_cast<float>(data[i]);
                if (std::isinf(val) || std::isnan(val)) {
                    overflow = true;
                    break;
                }
            }
        }
    });
    
    return overflow;
}

} // namespace ag

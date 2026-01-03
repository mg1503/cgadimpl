#include <torch/torch.h>
#include <torch/nn/modules/module.h>
#include <torch/nn/module/linear.h>
#include <torch/nn/modules/embedding.h>
#include <torch/nn/functions/activation.h>

class RMSNorm : public torch::nn::Module{
public:
RMSNorm(int64_t hidden_size, float eps = 1e-6f)
}
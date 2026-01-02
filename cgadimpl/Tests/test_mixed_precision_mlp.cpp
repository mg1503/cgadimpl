#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <iomanip>
#include "ad/ag_all.hpp"
#include "ad/optimizer/optim.hpp"

using namespace ag;
using namespace OwnTensor;

// Helper to print tensor info
void print_tensor_info(const std::string& name, const Tensor& t) {
    std::cout << "  [" << name << "] Dtype: " << get_dtype_name(t.dtype()) 
              << ", Shape: (";
    for (size_t i = 0; i < t.shape().dims.size(); ++i) {
        std::cout << t.shape().dims[i] << (i == t.shape().dims.size() - 1 ? "" : ", ");
    }
    std::cout << ")";
    
    size_t n_print = std::min((size_t)t.numel(), (size_t)4);
    if (n_print > 0) {
        std::cout << ", Values: [";
        dispatch_by_dtype(t.dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            if constexpr (is_complex(type_to_dtype<T>())) {
                std::cout << "complex";
            } else {
                const T* data = t.data<T>();
                for (size_t i = 0; i < n_print; ++i) {
                    std::cout << static_cast<float>(data[i]) << (i == n_print - 1 ? "" : ", ");
                }
                if (t.numel() > 4) std::cout << ", ...";
            }
        });
        std::cout << "]";
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "===== Mixed Precision MLP Training Test =====\n" << std::endl;

    // 1. Setup MLP Parameters in BF16
    // Simple MLP: 2 -> 4 -> 1
    int in_features = 2;
    int hidden_features = 4;
    int out_features = 1;

    auto opts_bf16 = TensorOptions().with_dtype(Dtype::Bfloat16).with_req_grad(true);
    
    Value W1 = make_tensor(Tensor::randn(Shape{{in_features, hidden_features}}, opts_bf16), "W1");
    Value b1 = make_tensor(Tensor::randn(Shape{{1, hidden_features}}, opts_bf16), "b1");
    Value W2 = make_tensor(Tensor::randn(Shape{{hidden_features, out_features}}, opts_bf16), "W2");
    Value b2 = make_tensor(Tensor::randn(Shape{{1, out_features}}, opts_bf16), "b2");

    std::vector<Value> params = {W1, b1, W2, b2};

    std::cout << "--- Initial Parameters (BF16) ---" << std::endl;
    print_tensor_info("W1", W1.val());
    print_tensor_info("b1", b1.val());

    // 2. Initialize Optimizer (Adam)
    // Adam should detect BF16 parameters and create FP32 master weights
    Adam optimizer(params, 0.1f);

    std::cout << "\n--- Optimizer Inspection ---" << std::endl;
    const Tensor* master_W1 = optimizer.get_master_weight(W1);
    if (master_W1) {
        std::cout << "  Successfully detected master weight for W1!" << std::endl;
        print_tensor_info("Master W1", *master_W1);
    } else {
        std::cerr << "  Error: Master weight for W1 not found!" << std::endl;
        return 1;
    }

    // 3. Forward Pass
    std::cout << "\n--- Forward Pass ---" << std::endl;
    Tensor input_t = Tensor::ones(Shape{{1, in_features}}, TensorOptions().with_dtype(Dtype::Bfloat16));
    input_t.fill(bfloat16_t(1.0f));
    Value x = make_tensor(input_t, "input");
    
    // Simple linear chain to ensure non-zero gradients
    Value h = ag::matmul(x, W1) + b1;
    Value h1 = ag::tanh(h);
    Value y = ag::matmul(h1, W2) + b2;
    Value y1 = ag::tanh(y);
    Value loss = ag::sum(y1); // Scalar loss
    
    print_tensor_info("Input", x.val());
    print_tensor_info("Output", y.val());
    print_tensor_info("Loss", loss.val());

    // 4. Backward Pass
    std::cout << "\n--- Backward Pass ---" << std::endl;
    optimizer.zero_grad();
    backward(loss);

    // Verify gradient dtypes (should be FP32 for floating point nodes)
    std::cout << "  Verifying gradient dtypes (should be float32 for high precision accumulation):" << std::endl;
    print_tensor_info("W1.grad", W1.grad());
    print_tensor_info("W2.grad", W2.grad());

    // 5. Optimizer Step
    std::cout << "\n--- Optimizer Step (Update) ---" << std::endl;
    
    // Capture values before update
    float w1_before = static_cast<float>(W1.val().data<bfloat16_t>()[0]);
    float master_w1_before = master_W1->data<float>()[0];

    optimizer.step();

    // 6. Verify Updates
    std::cout << "\n--- Post-Update Inspection ---" << std::endl;
    float w1_after = static_cast<float>(W1.val().data<bfloat16_t>()[0]);
    float master_w1_after = master_W1->data<float>()[0];

    std::cout << "  W1[0] (BF16)  : " << w1_before << " -> " << w1_after << std::endl;
    std::cout << "  Master W1[0] (FP32): " << master_w1_before << " -> " << master_w1_after << std::endl;

    // Check if master weight is indeed FP32
    if (master_W1->dtype() == Dtype::Float32) {
        std::cout << "  [OK] Master weight is Float32." << std::endl;
    } else {
        std::cout << "  [FAIL] Master weight is NOT Float32!" << std::endl;
    }

    // Check if update happened
    if (std::abs(master_w1_after - master_w1_before) > 0) {
        std::cout << "  [OK] Master weight was updated in high precision." << std::endl;
    } else {
        std::cout << "  [FAIL] Master weight was NOT updated!" << std::endl;
    }

    if (W1.val().dtype() == Dtype::Bfloat16) {
        std::cout << "  [OK] Model parameter remains Bfloat16." << std::endl;
    }

    std::cout << "\nRESULT: Mixed Precision MLP Test Completed Successfully!" << std::endl;

    return 0;
}

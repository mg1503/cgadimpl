#include "ad/ag_all.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>

using namespace ag;
using namespace std;
using namespace OwnTensor;

// Helper to print pass/fail
void assert_true(bool condition, const string& message) {
    if (condition) {
        cout << "[PASS] " << message << endl;
    } else {
        cout << "[FAIL] " << message << endl;
        exit(1);
    }
}

// Helper to check if gradient is computed (non-zero/non-empty)
void check_grad(Value& v, const string& name) {
    if (v.node && v.node->requires_grad()) {
        // Simple check: gradient should be allocated and have same shape
        bool has_grad = v.grad().numel() > 0;
        assert_true(has_grad, "Gradient computed for " + name);
    }
}

void test_activations() {
    cout << "\n--- Testing Activations ---\n";
    
    Tensor data = Tensor::randn<float>(Shape{{2, 3}}, TensorOptions().with_req_grad(true));
    Value x = make_tensor(data, "x");

    // 1. GELU
    {
        Value y = gelu(x);
        backward(sum(y)); // simple scalar backward
        check_grad(x, "GELU input");
        zero_grad(x);
        cout << "GELU test complete.\n";
    }

    // 2. ReLU
    {
        Value y = relu(x);
        backward(sum(y));
        check_grad(x, "ReLU input");
        zero_grad(x);
        cout << "ReLU test complete.\n";
    }

    // 3. Sigmoid
    {
        Value y = sigmoid(x);
        backward(sum(y));
        check_grad(x, "Sigmoid input");
        zero_grad(x);
        cout << "Sigmoid test complete.\n";
    }

    // 4. Softmax
    {
        Value y = softmax_row(x); // default axis -1
        backward(sum(y));
        check_grad(x, "Softmax input");
        zero_grad(x);
        cout << "Softmax test complete.\n";
    }

    // 5. Tanh
    {
        Value y = tanh(x);
        backward(sum(y));
        check_grad(x, "Tanh input");
        zero_grad(x);
        cout << "Tanh test complete.\n";
    }
}

void test_layers() {
    cout << "\n--- Testing Layers ---\n";

    // 1. Linear
    {
        Tensor x_data = Tensor::randn<float>(Shape{{2, 4}}, TensorOptions().with_req_grad(true));
        Value x = make_tensor(x_data, "x");
        nn::Linear linear(4, 2);
        
        Value y = linear(x);
        backward(sum(y));
        
        check_grad(x, "Linear input");
        // Check weights and bias grads
        for(auto& p : linear.parameters()) {
            check_grad(p, "Linear parameter");
        }
        cout << "Linear test complete.\n";
    }

    // Dropout
    {
        Tensor x_data = Tensor::randn<float>(Shape{{5, 5}}, TensorOptions().with_req_grad(true));
        Value x = make_tensor(x_data, "x");
        Tensor mask_data = Tensor::ones(x_data.shape(), TensorOptions());
        Value mask = make_tensor(mask_data, "mask");
        
        Value y = dropout(x, mask);
        backward(sum(y));
        check_grad(x, "Dropout input");
        cout << "Dropout test complete.\n";
    }

    // Flatten
    {
        Tensor x_data = Tensor::randn<float>(Shape{{2, 3, 4}}, TensorOptions().with_req_grad(true));
        Value x = make_tensor(x_data, "x");
        
        Value y = flatten(x);
        assert_true(y.shape().size() == 2, "Flatten output rank is 2");
        assert_true(y.shape()[0] == 2 && y.shape()[1] == 12, "Flatten output shape correct");
        
        backward(sum(y));
        check_grad(x, "Flatten input");
        cout << "Flatten test complete.\n";
    }
}

void test_losses() {
    cout << "\n--- Testing Losses ---\n";
    
    Tensor pred_data = Tensor::rand<float>(Shape{{2, 2}}, TensorOptions().with_req_grad(true)); // 0-1 for BCE
    Tensor target_data = Tensor::rand<float>(Shape{{2, 2}}, TensorOptions());
    
    Value pred = make_tensor(pred_data, "pred");
    Value target = make_tensor(target_data, "target");

    // 1. Binary Cross Entropy
    {
        Value loss = binary_cross_entropy(pred, target);
        backward(loss);
        check_grad(pred, "BCE input");
        zero_grad(pred);
        cout << "BCE test complete.\n";
    }

    // 2. Categorical Cross Entropy
    {
        // Preds should be probabilities summing to 1 for CCE usually, but for test we just check grad flow
        // Using softmax to ensure valid inputs
        Value sm_pred = softmax_row(pred); 
        Value loss = categorical_cross_entropy(sm_pred, target);
        backward(loss);
        check_grad(pred, "CCE input"); // Gradient should flow back to pred through softmax
        zero_grad(pred);
        cout << "CCE test complete.\n";
    }

    // 3. MAE Loss
    {
        Value loss = mae_loss(pred, target);
        backward(loss);
        check_grad(pred, "MAE input");
        zero_grad(pred);
        cout << "MAE test complete.\n";
    }

    // 4. MSE Loss
    {
        Value loss = mse_loss(pred, target);
        backward(loss);
        check_grad(pred, "MSE input");
        zero_grad(pred);
        cout << "MSE test complete.\n";
    }
}

int main() {
    try {
        test_activations();
        test_layers();
        test_losses();
        cout << "\nAll registered operations tests PASSED!\n";
    } catch (const std::exception& e) {
        cout << "\n[ERROR] Test failed with exception: " << e.what() << endl;
        return 1;
    }
    return 0;
}
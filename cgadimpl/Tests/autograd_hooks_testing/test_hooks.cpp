#include <iostream>
#include <vector>
#include <atomic>
#include <cassert>
#include <thread>
#include "ad/ag_all.hpp"
#include "nn/nn.hpp"

using namespace ag;

class DeepLinearModule : public nn::Module {
public:
    nn::Linear *l1, *l2, *l3;

    DeepLinearModule(int in_features, int hidden_features, int out_features, Device dev = Device::CPU) {
        l1 = new nn::Linear(in_features, hidden_features, dev);
        l2 = new nn::Linear(hidden_features, hidden_features, dev);
        l3 = new nn::Linear(hidden_features, out_features, dev);
        
        for (auto& p : l1->parameters()) params_.push_back(p);
        for (auto& p : l2->parameters()) params_.push_back(p);
        for (auto& p : l3->parameters()) params_.push_back(p);
    }

    Value operator()(Value x) override {
        return (*l3)((*l2)((*l1)(x)));
    }
};

void test_basic_hook() {
    std::cout << "Running test_basic_hook..." << std::endl;
    
    auto a = make_tensor(Tensor::randn<float>(Shape{{2, 2}}, TensorOptions().with_req_grad(true)), "a");
    auto b = make_tensor(Tensor::randn<float>(Shape{{2, 2}}, TensorOptions().with_req_grad(true)), "b");
    auto c = matmul(a, b);
    auto loss = sum(c);

    bool hook_called = false;
    a.register_hook([&](Node* n) {
        std::cout << "Hook called for node: " << n->debug_name << std::endl;
        hook_called = true;
    });

    backward(loss);

    if (hook_called) {
        std::cout << " test_basic_hook passed!" << std::endl;
    } else {
        std::cerr << " test_basic_hook failed: hook was not called!" << std::endl;
        exit(1);
    }
}

void test_module_hook() {
    std::cout << "\nRunning test_module_hook..." << std::endl;
    nn::Linear linear(10, 5);
    auto input = make_tensor(Tensor::randn<float>(Shape{{1, 10}}, TensorOptions()));
    auto output = linear(input);
    auto loss = sum(output);

    std::atomic<int> hooks_called{0};
    linear.register_backward_hook([&](Node* n) {
        std::cout << "Module hook called for parameter: " << n->debug_name << std::endl;
        hooks_called++;
    });

    backward(loss);

    if (hooks_called == 2) {
        std::cout << "test_module_hook passed!" << std::endl;
    } else {
        std::cerr << "test_module_hook failed: expected 2 hook calls, got " << hooks_called << "!" << std::endl;
        exit(1);
    }
}

void test_parallel_hook() {
    std::cout << "\nRunning test_parallel_hook..." << std::endl;
    
    nn::Linear linear(10, 5);
    auto input = make_tensor(Tensor::randn<float>(Shape{{1, 10}}, TensorOptions()));
    auto output = linear(input);
    auto loss = sum(output);

    std::atomic<int> hooks_called{0};
    linear.register_backward_hook([&](Node* n) {
        std::cout << "Parallel module hook called for parameter: " << n->debug_name << " on thread " << std::this_thread::get_id() << std::endl;
        hooks_called++;
    });

    zero_grad(loss);
    backward(loss, nullptr, true);

    if (hooks_called == 2) {
        std::cout << " test_parallel_hook passed!" << std::endl;
    } else {
        std::cerr << " test_parallel_hook failed: expected 2 hook calls in parallel, got " << hooks_called << "!" << std::endl;
        exit(1);
    }
}

void test_complex_module_hook() {
    std::cout << "\nRunning test_complex_module_hook..." << std::endl;
    
    int in_features = 10;
    int hidden_features = 8;
    int out_features = 5;
    Device dev = Device::CPU;
    
    DeepLinearModule model(in_features, hidden_features, out_features, dev);
    auto input = make_tensor(Tensor::randn<float>(Shape{{1, in_features}}, TensorOptions().with_device(dev)));
    auto output = model(input);
    auto loss = sum(output);

    std::atomic<int> hooks_called{0};
    model.register_backward_hook([&](Node* n) {
        std::cout << "Complex module hook called for node: " << n->debug_name << std::endl;
        hooks_called++;
    });

    backward(loss);

    // DeepLinearModule has 3 Linear layers.
    // Each Linear layer has 2 parameters (W and b).
    // Total parameters = 3 * 2 = 6.
    if (hooks_called == 6) {
        std::cout << "test_complex_module_hook passed!" << std::endl;
    } else {
        std::cerr << "test_complex_module_hook failed: expected 6 hook calls, got " << hooks_called << "!" << std::endl;
        exit(1);
    }
}

int main() {
    try {
        test_basic_hook();
        test_module_hook();
        test_parallel_hook();
        test_complex_module_hook();
        std::cout << "\nAll hook tests passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
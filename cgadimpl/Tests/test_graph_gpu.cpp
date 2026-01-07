

// #include "ad/ag_all.hpp"
// #include <cassert>

// using namespace ag;
// using namespace OwnTensor;

// // 1. Add a public member to the test model to store the last input Value
// class SimpleMLP : public nn::Module {
// public:
//     Value last_input; // Public member to hold the input Value

//     SimpleMLP(int in, int hidden, int out, Device dev) {
//         layers_.push_back(new nn::Linear(in, hidden, dev));
//         layers_.push_back(new nn::ReLU());
//         layers_.push_back(new nn::Linear(hidden, out, dev));
//         for (auto* mod : layers_) {
//             params_.insert(params_.end(), mod->parameters().begin(), mod->parameters().end());
//         }
//     }
//         using ag::nn::Module::operator();

//     Value operator()(Value x) override {
//         this->last_input = x; // Store the input Value
//         for (auto* layer : layers_) {
//             x = (*layer)(x);
//         }
//         return x;
//     }
// private:
//     std::vector<Module*> layers_;
// };

// void test_cuda_graph(Device device) {
//     if (device != Device::CUDA) return;
//     std::cout << "\n--- Testing CUDA Graph Capture ---\n";

//     SimpleMLP model(10, 20, 5, device);
//     Tensor x_tensor = Tensor::randn(Shape{{4, 10}}, TensorOptions().with_device(device));

//     // 2. Redesign the test to use the JIT compiler correctly
//     std::cout << "Performing warm-up forward pass...\n";
//     Value y_warmup = model(x_tensor);

//     std::cout << "Compiling the graph...\n";
//     auto plan = jit::compile(y_warmup, {model.last_input}, model.parameters());
    
//     std::vector<Tensor*> inputs = {&x_tensor};
//     std::vector<Tensor*> params;
//     for (auto& p : model.parameters()) {
//         params.push_back(&p.val());
//     }
//     Tensor y_out(y_warmup.val().shape(), ag::options(y_warmup.val()));

//     CudaGraphRunner runner;
//     std::cout << "Beginning capture of plan.run()...\n";
//     runner.begin_capture();
//     plan.run(inputs, params, y_out);
//     runner.end_capture();
//     std::cout << "Capture successful.\n";

//     std::cout << "Replaying graph...\n";
//     bool ok = runner.replay();
//     std::cout << "Replay " << (ok ? "successful" : "failed") << ".\n";
//     assert(ok);
// }

// int main() {
//     try {
//         #ifdef WITH_CUDA
//         test_cuda_graph(Device::CUDA);
//         #else
//         std::cout << "CUDA tests skipped (not compiled with CUDA support).\n";
//         #endif
//     } catch (const std::exception& e) {
//         std::cerr << "Caught exception: " << e.what() << std::endl;
//         return 1;
//     }
//     return 0;
// }



// #include "ad/ag_all.hpp"
// #include <cassert>

// using namespace ag;
// using namespace OwnTensor;

// // Use the exact same SimpleMLP you provided, as it's correct.
// class SimpleMLP : public nn::Module {
// public:
//     SimpleMLP(int in, int hidden, int out, Device dev) {
//         layers_.push_back(new nn::Linear(in, hidden, dev));
//         layers_.push_back(new nn::ReLU());
//         layers_.push_back(new nn::Linear(hidden, out, dev));
//         for (auto* mod : layers_) {
//             // FIX: Use the new public method to get parameters
//             for (auto& p : mod->parameters()) {
//                 params_.push_back(p);
//             }
//         }
//     }
//     using ag::nn::Module::operator();

//     Value operator()(Value x) override {
//         for (auto* layer : layers_) {
//             x = (*layer)(x);
//         }
//         return x;
//     }
// private:
//     std::vector<Module*> layers_;
// };

// void test_cuda_graph(Device device) {
//     if (device != Device::CUDA) return;
//     std::cout << "\n--- Testing CUDA Graph Capture (Directly) ---\n";

//     SimpleMLP model(10, 20, 5, device);
//     Tensor x_tensor = Tensor::randn(Shape{{4, 10}}, TensorOptions().with_device(device));

//     std::cout << "Performing warm-up forward passes (pre-allocates all memory)...\n";
    
//     // CRITICAL FIX: Run the model MULTIPLE times to ensure ALL intermediate
//     // tensors are allocated. The first run allocates, subsequent runs reuse.
//     for (int i = 0; i < 3; ++i) {
//         Value y_warmup = model(x_tensor);
//         cudaDeviceSynchronize();
//     }
    
//     std::cout << "All memory pre-allocated. Starting capture...\n";

//     CudaGraphRunner runner;
    
//     std::cout << "Beginning capture of the forward pass...\n";
//     runner.begin_capture();
    
//     // CRITICAL: This forward pass should NOT allocate any new memory.
//     // It reuses the memory allocated during warmup.
//     Value y_captured = model(x_tensor);
    
//     runner.end_capture();
//     std::cout << "Capture successful.\n";

//     std::cout << "Replaying graph...\n";
//     bool ok = runner.replay();
    
//     // Synchronize to ensure replay is complete
//     cudaDeviceSynchronize();

//     std::cout << "Replay " << (ok ? "successful" : "failed") << ".\n";
//     assert(ok);

//     std::cout << "CUDA Graph test completed successfully.\n";
// }

// int main() {
//     try {
//         #ifdef WITH_CUDA
//         test_cuda_graph(Device::CUDA);
//         #else
//         std::cout << "CUDA tests skipped (not compiled with CUDA support).\n";
//         #endif
//     } catch (const std::exception& e) {
//         std::cerr << "Caught exception: " << e.what() << std::endl;
//         return 1;
//     }
//     return 0;
// }



#include "ad/ag_all.hpp"
#include <cassert>

using namespace ag;
using namespace OwnTensor;

// Use the exact same SimpleMLP you provided, as it's correct.
class SimpleMLP : public nn::Module {
public:
    SimpleMLP(int in, int hidden, int out, Device dev) {
        layers_.push_back(new nn::Linear(in, hidden, dev));
        layers_.push_back(new nn::ReLU());
        layers_.push_back(new nn::Linear(hidden, out, dev));
        for (auto* mod : layers_) {
            // FIX: Use the new public method to get parameters
            for (auto& p : mod->parameters()) {
                params_.push_back(p);
            }
        }
    }
    using ag::nn::Module::operator();

    Value operator()(Value x) override {
        for (auto* layer : layers_) {
            x = (*layer)(x);
        }
        return x;
    }
private:
    std::vector<Module*> layers_;
};

void test_cuda_graph(Device device) {
    if (device != Device::CUDA) return;
    std::cout << "\n--- Testing CUDA Graph Capture (Directly) ---\n";

    SimpleMLP model(10, 20, 5, device);
    Tensor x_tensor = Tensor::randn<float>(Shape{{4, 10}}, TensorOptions().with_device(device));

    std::cout << "Performing warm-up forward passes (pre-allocates all memory)...\n";
    
    // CRITICAL FIX: Run the model MULTIPLE times to ensure ALL intermediate
    // tensors are allocated. The first run allocates, subsequent runs reuse.
    for (int i = 0; i < 3; ++i) {
        Value y_warmup = model(x_tensor);
        cudaDeviceSynchronize();
    }
    
    std::cout << "All memory pre-allocated. Starting capture...\n";

    CudaGraphRunner runner;
    
    std::cout << "Beginning capture of the forward pass...\n";
    runner.begin_capture();
    
    {
        // CRITICAL: This forward pass should NOT allocate any new memory.
        // It reuses the memory allocated during warmup.
        Value y_captured = model(x_tensor);
        
        runner.end_capture();
        std::cout << "Capture successful.\n";

        std::cout << "Replaying graph...\n";
        bool ok = runner.replay();
        
        // Synchronize to ensure replay is complete
        //cudaDeviceSynchronize();

        std::cout << "Replay " << (ok ? "successful" : "failed") << ".\n";
        assert(ok);

        // Ensure we free on the capture stream
        OwnTensor::cuda::setCurrentStream(runner.get_stream());
    }
    // Reset stream to default after y_captured is destroyed
    OwnTensor::cuda::setCurrentStream(nullptr);


    std::cout << "CUDA Graph test completed successfully.\n";
}

int main() {
    try {
        #ifdef WITH_CUDA
        test_cuda_graph(Device::CUDA);
        #else
        std::cout << "CUDA tests skipped (not compiled with CUDA support).\n";
        #endif
    } catch (const std::exception& e) {
        std::cerr << "Caught exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
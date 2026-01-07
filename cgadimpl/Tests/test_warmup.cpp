#include <iostream>
#include <chrono>
#include "ad/ag_all.hpp"

using namespace ag;

void bench(const std::string& label, Device device, bool use_dag) {
    TensorOptions opts = TensorOptions().with_device(device).with_req_grad(true);
    Shape s1{{2048, 1024}};
    Shape s2{{1024, 2048}};

    // Warm up the CUDA context and memory pool
    {
        Tensor tmp = Tensor::randn<float>(s1, opts);
        if (device == Device::CUDA) cudaDeviceSynchronize();
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    
    if (use_dag) {
        auto a = make_tensor(Tensor::randn<float>(s1, opts), "A");
        auto b = make_tensor(Tensor::randn<float>(s2, opts), "B");
        auto d = matmul(a, b);
    } else {
        Tensor A = Tensor::randn<float>(s1, opts);
        Tensor B = Tensor::randn<float>(s2, opts);
        Tensor D = matmul(A, B);
    }

    if (device == Device::CUDA) cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    
    float ms = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f;
    std::cout << label << ": " << ms << " ms" << std::endl;
}

int main() {
    #ifdef WITH_CUDA

    bench("1. Direct (First Run)", Device::CUDA, false);
    bench("2. With DAG (Second Run)", Device::CUDA, true);
    bench("3. Direct (Third Run)", Device::CUDA, false);
    bench("4. With DAG (Fourth Run)", Device::CUDA, true);


    #endif
    return 0;
}

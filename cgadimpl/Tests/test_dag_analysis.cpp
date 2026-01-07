#include <iostream>
#include <chrono>
#include "ad/ag_all.hpp"

using namespace ag;

void run_benchmark(const std::string& label, Device device) {
    std::cout << "\n=== Benchmarking on " << (device == Device::CPU ? "CPU" : "CUDA") << " [" << label << "] 10,000 Ops ===" <<std::endl;

    TensorOptions opts = TensorOptions().with_device(device).with_req_grad(true);
    Shape s1{{98,98}};
    Shape s2{{98,98}};
    Shape s3{{98, 98}};

    // 1. Direct Tensor Operations (No DAG)
    auto t1 = std::chrono::high_resolution_clock::now();
    Tensor A = Tensor::randn<float>(s1, opts);
    Tensor B = Tensor::randn<float>(s2, opts);
    Tensor C = Tensor::randn<float>(s3, opts);
    Tensor D = matmul(A, B);
    Tensor E = matmul(D, C);

    for (int i = 0; i < 1000; ++i) {
        D = matmul(A, B);
        E = matmul(D, C);
    }

    if (device == Device::CUDA) cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    float direct_ms = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f;

    // 2. Autograd operations (With DAG)
    auto t3 = std::chrono::high_resolution_clock::now();
    auto a = make_tensor(Tensor::randn<float>(s1, opts), "A");
    auto b = make_tensor(Tensor::randn<float>(s2, opts), "B");
    auto c = make_tensor(Tensor::randn<float>(s3, opts), "C");
    auto d = matmul(a, b);
    auto e = matmul(d, c);

    for (int i = 0; i < 1000; ++i) {
        d = matmul(a, b);
        e = matmul(d, c);
    }

    if (device == Device::CUDA) cudaDeviceSynchronize();
    auto t4 = std::chrono::high_resolution_clock::now();
    float dag_ms = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count() / 1000.0f;

    std::cout << "Direct (No DAG): " << direct_ms << " ms" << std::endl;
    std::cout << "With DAG Node:   " << dag_ms << " ms" << std::endl;
    // ag::debug::dump_dot(e, "graph_test_10000.jpg");
    
    // std::cout << "Overhead: " << (dag_ms - direct_ms) << " ms (" << (dag_ms / direct_ms) << "x)" << std::endl;
    
    // Summary and Validation
    // debug::print_dag_summary(e);
    // debug::validate_dag(e);
}

int main() {
    run_benchmark("Large Matrix", Device::CPU);
    
    #ifdef WITH_CUDA
    try {
        run_benchmark("Large Matrix", Device::CUDA);
    } catch (const std::exception& ex) {
        std::cerr << "CUDA benchmark failed: " << ex.what() << std::endl;
    }
    #endif

    return 0;
}

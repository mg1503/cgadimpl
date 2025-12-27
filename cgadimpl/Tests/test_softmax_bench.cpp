#include <torch/torch.h>
#include <chrono>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

#include "ad/ag_all.hpp"
#include "ad/utils/debug.hpp"
#include "ad/utils/export_hlo.hpp"

using namespace ag;
using namespace OwnTensor;

int main() {
    std::cout << "\n--- Softmax  Benchmark (Latency, Throughput, Accuracy) ---\n";
    auto run_softmax_benchmark = [&](int64_t N, int iters = 15) {
        std::cout << "\n[N=" << N << "] Benchmarking Softmax ...\n";

        // 1. Setup Data
        OwnTensor::Tensor Xt_bench(Shape{{1, N}}, false);
        float* x_ptr = Xt_bench.data<float>();
        for (int64_t i = 0; i < N; ++i) {
            x_ptr[i] = static_cast<float>(i % 100) / 50.0f - 1.0f; 
        }
        Value X_bench = make_tensor(Xt_bench, "X_bench");

        // 2. Accuracy Check
        Value L_ag = ag::softmax_row(X_bench);
        torch::Tensor x_torch = torch::from_blob(x_ptr, {N}, torch::kFloat32);
        torch::Tensor y_torch_ref = torch::softmax(x_torch,0);
        
        float max_err = 0.0f;
        double mse = 0.0;
        const float* ag_ptr = L_ag.val().data<float>();
        const float* torch_ptr = y_torch_ref.data_ptr<float>();
        
        for (int64_t i = 0; i < N; ++i) {
            float diff = std::abs(ag_ptr[i] - torch_ptr[i]);
            max_err = std::max(max_err, diff);
            mse += (double)diff * diff;
        }
        mse /= N;

        // 3. Latency & Throughput (AG)
        std::vector<double> ag_times;
        for (int i = 0; i < iters; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            Value tmp = ag::softmax_row(X_bench);
            auto t1 = std::chrono::high_resolution_clock::now();
            ag_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        std::sort(ag_times.begin(), ag_times.end());
        double ag_median = ag_times[iters / 2];

        // 4. Latency & Throughput (Torch)
        std::vector<double> torch_times;
        for (int i = 0; i < iters; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            torch::Tensor tmp = torch::softmax(x_torch,0);
            auto t1 = std::chrono::high_resolution_clock::now();
            torch_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        std::sort(torch_times.begin(), torch_times.end());
        double torch_median = torch_times[iters / 2];

        // 5. Results
        double ag_throughput = (double)N / (ag_median / 1000.0) / 1e6; // M elements/s
        double torch_throughput = (double)N / (torch_median / 1000.0) / 1e6;

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  AG Latency:     " << std::setw(10) << ag_median << " ms | Throughput: " << std::setw(10) << ag_throughput << " M elem/s\n";
        std::cout << "  Torch Latency:  " << std::setw(10) << torch_median << " ms | Throughput: " << std::setw(10) << torch_throughput << " M elem/s\n";
        std::cout << "  Accuracy (MSE): " << mse << " | Max Error: " << max_err << "\n";
    };

    std::vector<int64_t> sizes = {1000, 10000, 100000, 1000000, 10000000};
    for (auto n : sizes) {
        run_softmax_benchmark(n);
    }

    return 0;
}
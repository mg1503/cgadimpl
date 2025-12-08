#include <cblas.h>
#include <chrono>
#include <iostream>
#include <vector>

void benchmark_openblas(int size) {
    std::vector<float> A(size * size, 1.0f);
    std::vector<float> B(size * size, 2.0f); 
    std::vector<float> C(size * size, 0.0f);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // sgemm: single-precision general matrix multiplication
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                size, size, size, 1.0f, 
                A.data(), size, B.data(), size, 0.0f, C.data(), size);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "OpenBLAS " << size << "x" << size << ": " 
              << duration.count() / 1000.0 << " ms" << std::endl;
    
    // Verify result (should be 1024.0 for 512x512 with 1.0 and 2.0)
    std::cout << "Result check: " << C[0] << std::endl;
}

int main() {
    benchmark_openblas(4096);
    return 0;
}
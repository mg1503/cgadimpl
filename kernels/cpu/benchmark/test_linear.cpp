// // test_linear.cpp
// // Compile with: -O3 -mavx2 -mfma -fopenmp  -I/path/to/eigen
// #include <iostream>
// #include <vector>
// #include <chrono>
// #include <random>
// #include <cmath>
// #include <Eigen/Dense>

// // declare the optimized kernels (from linear_kernels.cpp)
// extern "C" {
//     void linear_impl_optimized(const float* X, const float* W, const float* b,
//                                float* Y, int B, int In, int Out);
//     void linear_dW_impl_optimized(const float* X, const float* dY, float* dW,
//                                   int B, int In, int Out);
//     void linear_dX_impl_optimized(const float* dY, const float* W, float* dX,
//                                   int B, int In, int Out);
//     void linear_db_impl_optimized(const float* dY, float* db, int B, int Out);
// }

// static void fill_random(std::vector<float>& v, unsigned seed = 42) {
//     std::mt19937 gen(seed);
//     std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
//     for (auto &x : v) x = dist(gen);
// }

// static bool allclose_vec(const std::vector<float>& a, const std::vector<float>& b, float tol=1e-4f) {
//     if (a.size() != b.size()) return false;
//     double err = 0.0;
//     for (size_t i=0;i<a.size();++i) {
//         double d = std::fabs((double)a[i] - (double)b[i]);
//         if (d > tol) return false;
//         err += d;
//     }
//     return true;
// }

// int main() {
//     std::cout << "===== Optimized Linear (forward+backward) Test =====\n";

//     const int B = 32;
//     const int In = 37;   // intentionally not multiple of 8
//     const int Out = 61;  // intentionally not multiple of 8
//     const int RUNS = 50;

//     std::vector<float> X((size_t)B*In);
//     std::vector<float> W((size_t)In*Out);
//     std::vector<float> b(Out);
//     std::vector<float> Y_opt((size_t)B*Out), Y_ref((size_t)B*Out);

//     fill_random(X, 123);
//     fill_random(W, 456);
//     fill_random(b, 789);

//     // Eigen maps (row-major)
//     Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> X_eig(X.data(), B, In);
//     Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> W_eig(W.data(), In, Out);
//     Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Y_eig_ref(Y_ref.data(), B, Out);
//     Eigen::VectorXf b_eig = Eigen::Map<Eigen::VectorXf>(b.data(), Out);

//     // Reference forward
//     Y_eig_ref.noalias() = X_eig * W_eig;
//     Y_eig_ref.rowwise() += b_eig.transpose();

//     // optimized forward timing
//     double t_opt = 0.0;
//     for (int r=0;r<RUNS;++r) {
//         auto t0 = std::chrono::high_resolution_clock::now();
//         linear_impl_optimized(X.data(), W.data(), b.data(), Y_opt.data(), B, In, Out);
//         auto t1 = std::chrono::high_resolution_clock::now();
//         t_opt += std::chrono::duration<double,std::milli>(t1 - t0).count();
//     }
//     t_opt /= RUNS;

//     // Eigen timing
//     double t_eig = 0.0;
//     for (int r=0;r<RUNS;++r) {
//         auto t0 = std::chrono::high_resolution_clock::now();
//         Y_eig_ref.noalias() = X_eig * W_eig;
//         Y_eig_ref.rowwise() += b_eig.transpose();
//         auto t1 = std::chrono::high_resolution_clock::now();
//         t_eig += std::chrono::duration<double,std::milli>(t1 - t0).count();
//     }
//     t_eig /= RUNS;

//     std::cout << "Forward: optimized vs Eigen\n";
//     bool okf = allclose_vec(Y_opt, Y_ref);
//     std::cout << (okf ? "  ✅ outputs match\n" : "  ❌ outputs mismatch\n");
//     if (!okf) {
//         for (int i=0;i<8 && i<(int)Y_opt.size();++i)
//             std::cout << "  i="<<i<<" opt="<<Y_opt[i]<<" ref="<<Y_ref[i]<<"\n";
//     }
//     std::cout << "  opt time (ms) = " << t_opt << " | eig time (ms) = " << t_eig
//               << " | ratio opt/eig = " << (t_opt / t_eig) << "\n\n";

//     // -------------------------
//     // Backward: create random dY and compare dW,dX,db
//     // -------------------------
//     std::vector<float> dY((size_t)B*Out);
//     fill_random(dY, 999);

//     // Reference grads (Eigen)
//     Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> dY_eig(dY.data(), B, Out);
//     Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dW_ref(In, Out);
//     Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dX_ref(B, In);
//     Eigen::VectorXf db_ref(Out);

//     // dW_ref = X^T * dY
//     dW_ref.noalias() = X_eig.transpose() * dY_eig;
//     // dX_ref = dY * W^T
//     dX_ref.noalias() = dY_eig * W_eig.transpose();
//     // db_ref = sum rows of dY
//     db_ref = dY_eig.colwise().sum();

//     // optimized outputs
//     std::vector<float> dW_opt((size_t)In * Out);
//     std::vector<float> dX_opt((size_t)B * In);
//     std::vector<float> db_opt(Out);

//     // time optimized backward (single run for each)
//     auto t0 = std::chrono::high_resolution_clock::now();
//     linear_dW_impl_optimized(X.data(), dY.data(), dW_opt.data(), B, In, Out);
//     linear_dX_impl_optimized(dY.data(), W.data(), dX_opt.data(), B, In, Out);
//     linear_db_impl_optimized(dY.data(), db_opt.data(), B, Out);
//     auto t1 = std::chrono::high_resolution_clock::now();
//     double t_back_opt = std::chrono::duration<double,std::milli>(t1 - t0).count();

//     // Convert refs to vectors for comparison
//     std::vector<float> dW_ref_v((size_t)In*Out), dX_ref_v((size_t)B*In), db_ref_v(Out);
//     for (int i=0;i<In;++i) for (int j=0;j<Out;++j) dW_ref_v[(size_t)i*Out + j] = dW_ref(i,j);
//     for (int b=0;b<B;++b) for (int k=0;k<In;++k) dX_ref_v[(size_t)b*In + k] = dX_ref(b,k);
//     for (int j=0;j<Out;++j) db_ref_v[j] = db_ref(j);

//     bool ok_dW = allclose_vec(dW_opt, dW_ref_v, 1e-3f);
//     bool ok_dX = allclose_vec(dX_opt, dX_ref_v, 1e-3f);
//     bool ok_db = allclose_vec(db_opt, db_ref_v, 1e-6f);

//     std::cout << "Backward correctness:\n";
//     std::cout << "  dW match: " << (ok_dW ? "✅" : "❌") << "\n";
//     std::cout << "  dX match: " << (ok_dX ? "✅" : "❌") << "\n";
//     std::cout << "  db  match: " << (ok_db ? "✅" : "❌") << "\n";
//     if (!ok_dW) {
//         std::cout << "First few dW diffs:\n";
//         for (int i=0;i<std::min(8, In); ++i)
//             std::cout << " dW_opt[0.." << i << "] = " << dW_opt[i] << " ref=" << dW_ref_v[i] << "\n";
//     }

//     std::cout << "Backward optimized time (single run) = " << t_back_opt << " ms\n";

//     std::cout << "\n===== Test finished =====\n";
//     return 0;
// }


#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <Eigen/Dense>
#include "ad/kernels_api.hpp"

using namespace std;

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

void fill_random(vector<float>& v, float scale = 1.0f) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(gen) * scale;
}

bool allclose(const vector<float>& a, const vector<float>& b, float tol = 1e-4f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (fabs(a[i] - b[i]) > tol)
            return false;
    return true;
}

float mean_abs_diff(const vector<float>& a, const vector<float>& b) {
    float s = 0.f;
    for (size_t i = 0; i < a.size(); ++i)
        s += fabs(a[i] - b[i]);
    return s / a.size();
}

// ----------------------------------------------------------------------
// Global test runner
// ----------------------------------------------------------------------

int main() {
    cout << "===== Optimized CPU Kernels: Comprehensive Test =====\n";

    // Load plugin
    ag_cpu_v1 cpu{};
    if (ag_get_cpu_kernels_v1(&cpu) != 0) {
        cerr << "❌ Failed to load kernels via ag_get_cpu_kernels_v1()\n";
        return 1;
    }
    cout << "✅ Kernels loaded.\n";

    // Random data dimensions
    const int B = 32;     // batch
    const int In = 37;
    const int Out = 61;
    const int N_RUNS = 100;

    vector<float> X(B * In);
    vector<float> W(In * Out);
    vector<float> b(Out);
    vector<float> Y_opt(B * Out);
    vector<float> Y_ref(B * Out);

    fill_random(X);
    fill_random(W);
    fill_random(b);

    // ------------------------------------------------------------------
    // Linear Forward Test vs Eigen
    // ------------------------------------------------------------------
    cout << "\n--- [1] Linear Forward ---\n";

    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        X_eig(X.data(), B, In);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        W_eig(W.data(), In, Out);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        Y_eig_ref(Y_ref.data(), B, Out);
    Eigen::VectorXf b_eig = Eigen::Map<Eigen::VectorXf>(b.data(), Out);

    Y_eig_ref.noalias() = X_eig * W_eig;
    Y_eig_ref.rowwise() += b_eig.transpose();

    // Optimized
    auto t0 = chrono::high_resolution_clock::now();
    for (int i = 0; i < N_RUNS; ++i)
        cpu.linear(X.data(), W.data(), b.data(), Y_opt.data(), B, In, Out);
    auto t1 = chrono::high_resolution_clock::now();

    double t_linear = chrono::duration<double, milli>(t1 - t0).count() / N_RUNS;

    bool ok_linear = allclose(Y_opt, Y_ref);
    cout << (ok_linear ? "✅ Matches Eigen.\n" : "❌ Mismatch!\n");
    cout << "Avg time: " << t_linear << " ms\n";

    // ------------------------------------------------------------------
    // Elementwise Forward/Backward (ReLU, Sigmoid, Tanh, etc.)
    // ------------------------------------------------------------------
    cout << "\n--- [2] Elementwise Activation Tests ---\n";
    const int N = 512;
    vector<float> in(N), out(N), grad_in(N), grad_out(N);
    fill_random(in);

    // Forward tests
    cout << "ReLU forward: ";
    cpu.relu(in.data(), out.data(), N);
    cout << "✅\n";

    cout << "Sigmoid forward: ";
    cpu.sigmoid(in.data(), out.data(), N);
    cout << "✅\n";

    cout << "Tanh forward: ";
    cpu.tanh(in.data(), out.data(), N);
    cout << "✅\n";

    cout << "Softplus forward: ";
    cpu.softmax(in.data(), out.data(), N);
    cout << "✅\n";

    // Backward tests
    cout << "\nBackward sanity checks:\n";
    fill_random(grad_out);
    cpu.relu_bwd(in.data(), grad_out.data(), grad_in.data(), N);
    cout << "ReLU backward ✅\n";
    cpu.sigmoid_bwd_from_s(out.data(), grad_out.data(), grad_in.data(), N);
    cout << "Sigmoid backward ✅\n";
    cpu.tanh_bwd_from_t(out.data(), grad_out.data(), grad_in.data(), N);
    cout << "Tanh backward ✅\n";

    // ------------------------------------------------------------------
    // Linear Backward (dW, dX, db)
    // ------------------------------------------------------------------
    cout << "\n--- [3] Linear Backward ---\n";

    vector<float> dY(B * Out), dW(In * Out, 0.0f), dX(B * In, 0.0f), db(Out, 0.0f);
    fill_random(dY);

    // Compute gradients using optimized kernels
    auto t2 = chrono::high_resolution_clock::now();
    for (int i = 0; i < N_RUNS; ++i) {
        fill(dW.begin(), dW.end(), 0.0f);
        fill(dX.begin(), dX.end(), 0.0f);
        fill(db.begin(), db.end(), 0.0f);
        cpu.linear_dW(X.data(), dY.data(), dW.data(), B, In, Out);
        cpu.linear_dX(dY.data(), W.data(), dX.data(), B, In, Out);
        cpu.linear_db(dY.data(), db.data(), B, Out);
    }
    auto t3 = chrono::high_resolution_clock::now();

    double t_back = chrono::duration<double, milli>(t3 - t2).count() / N_RUNS;
    cout << "✅ Linear backward ran.\n";
    cout << "Avg time: " << t_back << " ms\n";

    // Compare dW with Eigen (for correctness)
    Eigen::MatrixXf dW_ref = X_eig.transpose() * Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(dY.data(), B, Out);
    vector<float> dW_ref_v(dW_ref.data(), dW_ref.data() + In * Out);
    float diff = mean_abs_diff(dW, dW_ref_v);
    cout << "dW mean abs diff = " << diff << "\n";

    cout << "===== All Kernel Tests Completed =====\n";
    return 0;
}

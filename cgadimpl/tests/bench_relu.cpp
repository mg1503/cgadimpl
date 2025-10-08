// bench_relu.cpp
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "ad/kernels_api.hpp"  // from cgadimpl/include

// --------------------- reference (your old/normal ReLU) ---------------------
static void relu_ref(const float* x, float* y, int64_t n) {
  for (int64_t i = 0; i < n; ++i) y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

// --------------------- timing helpers ---------------------
using clock_type = std::chrono::steady_clock;

static double time_ms(void(*fn)(const float*, float*, int64_t),
                      const float* x, float* y, int64_t n, int iters) {
  // Warm-up
  for (int i = 0; i < 3; ++i) fn(x, y, n);

  std::vector<double> times; times.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    auto t0 = clock_type::now();
    fn(x, y, n);
    auto t1 = clock_type::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    times.push_back(ms);
  }
  std::sort(times.begin(), times.end());
  // Return median
  return times[times.size() / 2];
}

static void fill_random(float* p, int64_t n, uint32_t seed=123) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
  for (int64_t i = 0; i < n; ++i) p[i] = dist(rng);
}

static bool arrays_equal(const float* a, const float* b, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    if (a[i] != b[i]) return false; // exact compare fine for ReLU
  }
  return true;
}

int main(int argc, char** argv) {
  // Problem size: default 64M elements (~256MB in, 256MB out)
  int64_t n = (argc > 1) ? std::stoll(argv[1]) : (1LL << 26);
  int iters = (argc > 2) ? std::stoi(argv[2]) : 15;

  // Load plugin (adjust path)
  const char* libpath = (argc > 3) ? argv[3] : "./libagkernels_cpu.so";
  try {
    ag::kernels::load_cpu_plugin(libpath);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Failed to load plugin: %s\n", e.what());
    return 1;
  }

  auto* relu_kernel = ag::kernels::cpu().relu;
  if (!relu_kernel) {
    std::fprintf(stderr, "Plugin does not provide ReLU kernel.\n");
    return 1;
  }

  // Allocate
  std::vector<float> x(n), y_ref(n), y_ker(n);
  fill_random(x.data(), n, 123);

  // Correctness check
  relu_ref(x.data(), y_ref.data(), n);
  relu_kernel(x.data(), y_ker.data(), n);
  if (!arrays_equal(y_ref.data(), y_ker.data(), n)) {
    std::fprintf(stderr, "Mismatch between reference and kernel outputs!\n");
    return 2;
  }

  // Timings
  double ms_ref = time_ms(relu_ref, x.data(), y_ref.data(), n, iters);
  double ms_ker = time_ms(relu_kernel, x.data(), y_ker.data(), n, iters);

  // Throughput metrics
  // Each ReLU reads x and writes y → ~8 bytes/element (read + write)
  double gb = (double)n * 8.0 / (1024.0*1024.0*1024.0);
  double ref_gbps = gb / (ms_ref / 1000.0);
  double ker_gbps = gb / (ms_ker / 1000.0);
  double ref_gelms = (double)n / (ms_ref / 1000.0) / 1e9; // G elems/s
  double ker_gelms = (double)n / (ms_ker / 1000.0) / 1e9;

  std::printf("N=%lld  iters=%d\n", (long long)n, iters);
  std::printf("Reference ReLU : %8.3f ms | %6.2f GB/s | %6.2f G elems/s\n",
              ms_ref, ref_gbps, ref_gelms);
  std::printf("Plugin    ReLU : %8.3f ms | %6.2f GB/s | %6.2f G elems/s\n",
              ms_ker, ker_gbps, ker_gelms);
  std::printf("Speedup        : %8.3fx\n", ms_ref / ms_ker);

  return 0;
}

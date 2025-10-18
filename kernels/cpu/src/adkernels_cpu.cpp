// cgadimpl/kernels/cpu/src/adkernels_cpu.cpp
#include "ad/kernels_api.hpp"
#include "matker.cuh"
#include <immintrin.h>
#include <omp.h>
#include <cstdint>
#include <cmath>
#include <cstddef>
#include <cassert>
#include <vector>
#include <cstring>
#include <algorithm>

extern "C" {

// ReLU backward: dX = dY * (x > 0 ? 1 : 0)
void relu_bwd_impl_optimized(const float* x, const float* dY, float* dX, int64_t n) {
    const __m256 zero = _mm256_setzero_ps();
    #pragma omp parallel for
    for (int64_t i = 0; i < n; i += 8) {
        if (i + 8 <= n) {
            __m256 xv = _mm256_loadu_ps(x + i);
            __m256 dyv = _mm256_loadu_ps(dY + i);
            __m256 mask = _mm256_cmp_ps(xv, zero, _CMP_GT_OS); // true where x>0
            __m256 res = _mm256_and_ps(dyv, mask); // keep dY where mask true
            _mm256_storeu_ps(dX + i, res);
        } else {
            for (int64_t j = i; j < n; ++j) dX[j] = x[j] > 0.0f ? dY[j] : 0.0f;
        }
    }
}

// LeakyReLU backward: y = (x > 0) ? x : alpha*x
// dX = dY * (x > 0 ? 1 : alpha)
void leakyrelu_bwd_impl_optimized(const float* x, const float* dY, float* dX, int64_t n, float alpha) {
    const __m256 zero = _mm256_setzero_ps();
    const __m256 aval = _mm256_set1_ps(alpha);
    #pragma omp parallel for
    for (int64_t i = 0; i < n; i += 8) {
        if (i + 8 <= n) {
            __m256 xv = _mm256_loadu_ps(x + i);
            __m256 dy = _mm256_loadu_ps(dY + i);
            __m256 mask = _mm256_cmp_ps(xv, zero, _CMP_GT_OS); // x>0
            __m256 neg_mult = _mm256_mul_ps(dy, aval);         // alpha * dY
            __m256 res = _mm256_blendv_ps(neg_mult, dy, mask); // choose dY or alpha*dY
            _mm256_storeu_ps(dX + i, res);
        } else {
            for (int64_t j = i; j < n; ++j) dX[j] = x[j] > 0.0f ? dY[j] : alpha * dY[j];
        }
    }
}

// Sigmoid backward: s = sigmoid(x); dX = dY * s * (1 - s)
// If forward stored sigmoid output 's' instead of x, you can accept s directly.
void sigmoid_bwd_impl_optimized_from_x(const float* x, const float* dY, float* dX, int64_t n) {
    // compute sigmoid(x) then derivative
    const __m256 one = _mm256_set1_ps(1.0f);
    #pragma omp parallel for
    for (int64_t i = 0; i < n; i += 8) {
        if (i + 8 <= n) {
            __m256 xv = _mm256_loadu_ps(x + i);
            // sigmoid = 1/(1+exp(-x)). Use approximate exp via standard scalar for simplicity.
            // For speed, if you have exp256_approx, call it here.
            // We'll use _mm256_exp_ps if available; otherwise, fall back to scalar per lane.
            float tmp[8]; _mm256_storeu_ps(tmp, xv);
            float s_arr[8];
            for (int k = 0; k < 8; ++k) s_arr[k] = 1.0f / (1.0f + std::exp(-tmp[k]));
            __m256 s = _mm256_loadu_ps(s_arr);
            __m256 dy = _mm256_loadu_ps(dY + i);
            __m256 t = _mm256_mul_ps(s, _mm256_sub_ps(one, s)); // s*(1-s)
            __m256 res = _mm256_mul_ps(dy, t);
            _mm256_storeu_ps(dX + i, res);
        } else {
            for (int64_t j = i; j < n; ++j) {
                float s = 1.0f / (1.0f + std::exp(-x[j]));
                dX[j] = dY[j] * s * (1.0f - s);
            }
        }
    }
}

// If you stored sigmoid output s in forward, you can implement a faster version:
void sigmoid_bwd_impl_optimized_from_s(const float* s, const float* dY, float* dX, int64_t n) {
    const __m256 one = _mm256_set1_ps(1.0f);
    #pragma omp parallel for
    for (int64_t i = 0; i < n; i += 8) {
        if (i + 8 <= n) {
            __m256 sv = _mm256_loadu_ps(s + i);
            __m256 dy = _mm256_loadu_ps(dY + i);
            __m256 t = _mm256_mul_ps(sv, _mm256_sub_ps(one, sv)); // s*(1-s)
            __m256 res = _mm256_mul_ps(dy, t);
            _mm256_storeu_ps(dX + i, res);
        } else {
            for (int64_t j = i; j < n; ++j)
                dX[j] = dY[j] * s[j] * (1.0f - s[j]);
        }
    }
}

// Tanh backward: t = tanh(x); dX = dY * (1 - t^2)
// If forward stored tanh(x) as 't', use that for faster compute.
void tanh_bwd_impl_optimized_from_t(const float* t, const float* dY, float* dX, int64_t n) {
    const __m256 one = _mm256_set1_ps(1.0f);
    #pragma omp parallel for
    for (int64_t i = 0; i < n; i += 8) {
        if (i + 8 <= n) {
            __m256 tv = _mm256_loadu_ps(t + i);
            __m256 dy = _mm256_loadu_ps(dY + i);
            __m256 tv2 = _mm256_mul_ps(tv, tv);
            __m256 tterm = _mm256_sub_ps(one, tv2);
            __m256 res = _mm256_mul_ps(dy, tterm);
            _mm256_storeu_ps(dX + i, res);
        } else {
            for (int64_t j = i; j < n; ++j)
                dX[j] = dY[j] * (1.0f - t[j]*t[j]);
        }
    }
}

// GELU backward using tanh-approx derivative:
// For GELU(x) = 0.5*x*(1 + tanh(u)), u = sqrt(2/pi)*(x + 0.044715 x^3)
// derivative (from common approximation) implemented elementwise:
void gelu_bwd_impl_optimized(const float* x, const float* dY, float* dX, int64_t n) {
    const __m256 kSqrt2OverPi = _mm256_set1_ps(0.7978845608028654f);
    const __m256 k0_044715 = _mm256_set1_ps(0.044715f);
    const __m256 k0_5 = _mm256_set1_ps(0.5f);
    const __m256 one = _mm256_set1_ps(1.0f);
    #pragma omp parallel for
    for (int64_t i = 0; i < n; i += 8) {
        if (i + 8 <= n) {
            __m256 xv = _mm256_loadu_ps(x + i);
            // x^2, x^3
            __m256 x2 = _mm256_mul_ps(xv, xv);
            __m256 x3 = _mm256_mul_ps(x2, xv);
            // u = k * (x + a x^3)
            __m256 term = _mm256_fmadd_ps(k0_044715, x3, xv);
            __m256 u = _mm256_mul_ps(term, kSqrt2OverPi);
            // tanh(u) approx as earlier: use rational approx
            __m256 u2 = _mm256_mul_ps(u, u);
            __m256 num = _mm256_add_ps(_mm256_set1_ps(27.0f), u2);       // 27 + u^2
            __m256 den = _mm256_fmadd_ps(_mm256_set1_ps(9.0f), u2, _mm256_set1_ps(27.0f)); // 9u^2 + 27
            __m256 th = _mm256_div_ps(_mm256_mul_ps(u, num), den); // approx tanh(u)
            // derivative pieces: y = 0.5 * x * (1 + th)
            // dy/dx = 0.5*(1 + th) + 0.5*x * (1 - th^2) * du/dx
            // du/dx = k * (1 + 3*a*x^2)
            __m256 one_plus_th = _mm256_add_ps(one, th);
            __m256 term2 = _mm256_sub_ps(one, _mm256_mul_ps(th, th)); // 1 - th^2
            __m256 three_ax2 = _mm256_fmadd_ps(_mm256_set1_ps(3.0f * 0.044715f), x2, _mm256_set1_ps(1.0f));
            __m256 dudx = _mm256_mul_ps(kSqrt2OverPi, three_ax2);
            __m256 part1 = _mm256_mul_ps(k0_5, one_plus_th); // 0.5*(1+th)
            __m256 part2 = _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(k0_5, xv), term2), dudx);
            __m256 grad = _mm256_mul_ps(_mm256_add_ps(part1, part2), _mm256_loadu_ps(dY + i));
            _mm256_storeu_ps(dX + i, grad);
        } else {
            for (int64_t j = i; j < n; ++j) {
                float v = x[j];
                float v2 = v*v;
                float v3 = v2*v;
                float u = 0.7978845608028654f * (v + 0.044715f * v3);
                float th = std::tanh(u);
                float one_plus_th = 1.0f + th;
                float term2 = 1.0f - th*th;
                float dudx = 0.7978845608028654f * (1.0f + 3.0f * 0.044715f * v2);
                float part1 = 0.5f * one_plus_th;
                float part2 = 0.5f * v * term2 * dudx;
                dX[j] = (part1 + part2) * dY[j];
            }
        }
    }
}

// Softplus backward: d/dx log(1+exp(x)) = sigmoid(x)
void softplus_bwd_impl_optimized(const float* x, const float* dY, float* dX, int64_t n) {
    // use sigmoid(x) as derivative
    const __m256 one = _mm256_set1_ps(1.0f);
    #pragma omp parallel for
    for (int64_t i = 0; i < n; i += 8) {
        if (i + 8 <= n) {
            float tmp_x[8]; _mm256_storeu_ps(tmp_x, _mm256_loadu_ps(x + i));
            float s[8];
            for (int k = 0; k < 8; ++k) s[k] = 1.0f / (1.0f + std::exp(-tmp_x[k]));
            __m256 sv = _mm256_loadu_ps(s);
            __m256 dy = _mm256_loadu_ps(dY + i);
            __m256 res = _mm256_mul_ps(dy, sv);
            _mm256_storeu_ps(dX + i, res);
        } else {
            for (int64_t j = i; j < n; ++j) {
                float s = 1.0f / (1.0f + std::exp(-x[j]));
                dX[j] = dY[j] * s;
            }
        }
    }
}

// Exp backward: d/dx exp(x) = exp(x); dX = dY * exp(x)
void exp_bwd_impl_optimized_from_y(const float* y, const float* dY, float* dX, int64_t n) {
    // if forward stored y = exp(x), this is fastest
    #pragma omp parallel for
    for (int64_t i = 0; i < n; i += 8) {
        if (i + 8 <= n) {
            __m256 yv = _mm256_loadu_ps(y + i);
            __m256 dy = _mm256_loadu_ps(dY + i);
            __m256 res = _mm256_mul_ps(dy, yv);
            _mm256_storeu_ps(dX + i, res);
        } else {
            for (int64_t j = i; j < n; ++j) dX[j] = dY[j] * y[j];
        }
    }
}

// Log backward: d/dx log(x) = 1/x; dX = dY / x
void log_bwd_impl_optimized(const float* x, const float* dY, float* dX, int64_t n) {
    #pragma omp parallel for
    for (int64_t i = 0; i < n; i += 8) {
        if (i + 8 <= n) {
            __m256 xv = _mm256_loadu_ps(x + i);
            __m256 dy = _mm256_loadu_ps(dY + i);
            __m256 res = _mm256_div_ps(dy, xv);
            _mm256_storeu_ps(dX + i, res);
        } else {
            for (int64_t j = i; j < n; ++j) dX[j] = dY[j] / x[j];
        }
    }
}

// Sqrt backward: y = sqrt(x) ; d/dx sqrt(x) = 1/(2*sqrt(x)) ; if forward stored y you can use y.
void sqrt_bwd_impl_optimized_from_y(const float* y, const float* dY, float* dX, int64_t n) {
    const __m256 two = _mm256_set1_ps(2.0f);
    #pragma omp parallel for
    for (int64_t i = 0; i < n; i += 8) {
        if (i + 8 <= n) {
            __m256 yv = _mm256_loadu_ps(y + i);
            __m256 dy = _mm256_loadu_ps(dY + i);
            __m256 denom = _mm256_mul_ps(two, yv);
            __m256 res = _mm256_div_ps(dy, denom);
            _mm256_storeu_ps(dX + i, res);
        } else {
            for (int64_t j = i; j < n; ++j) dX[j] = dY[j] / (2.0f * y[j]);
        }
    }
}
// Compute dA = dC @ B^T
// A: [M,K], B: [K,N], dC: [M,N]
void matmul_bwd_dA_impl_optimized(const float* dC, const float* B, float* dA, int M, int K, int N) {
    // dA = dC @ B^T  -> shapes (M,N) @ (N,K) -> (M,K)
    // If your gemm_impl_optimized has signature gemm(A,B,C_out,M,K,N) where C_out=MxN
    // we call gemm with params (dC, B, dA, M, N, K) and interpret shapes accordingly.
    // Use your optimized GEMM (gemm_impl_optimized) here:
    // gemm_impl_optimized(A, B, C_out, M, K, N);  // original form
    // To compute dA: A=dC (M,N), B=B (N,K) -> result (M,K)
    extern void gemm_impl_optimized(const float* A, const float* B, const float* C_in_unused, float* Out, int M_in, int K_in, int N_in);
    // call: A=dC, B = <B transposed?>  if gemm expects B as (K,N) -> adjust
    // simplest: implement a small routine that computes dA with existing matmul_impl_cudatile if that supports any shape.
    // Here we assume we can call matmul_impl_cudatile(dC, B, dA, M, N, K) where B is accessed as (N,K).
    extern void matmul_impl_cudatile(const float* A, const float* B, float* C, int M, int K, int N);
    matmul_impl_cudatile(dC, B, dA, M, N, K);
}

// Compute dB = A^T @ dC
// A: [M,K], dC: [M,N] -> A^T: [K,M] @ [M,N] = [K,N]
void matmul_bwd_dB_impl_optimized(const float* A, const float* dC, float* dB, int M, int K, int N) {
    // dB = A^T @ dC -> (K,M) @ (M,N) -> (K,N)
    extern void matmul_impl_cudatile(const float* A, const float* B, float* C, int M, int K, int N);
    // pass A^T by pointing to A with interpreter that matmul_impl_cudatile expects A as (M,K). Many libraries require explicit transpose.
    // If matmul_impl_cudatile cannot accept transposed inputs, you can compute dB in blocks or implement a transposed GEMM wrapper.
    // Here we'll do a simple approach: loop over k and n computing dot-products (works but slower). Ideally call transpose-capable gemm.
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            const float* Acol = A + k; // stride K? careful: A is row-major: A[i*K + k]
            for (int m = 0; m < M; ++m) {
                sum += A[m*K + k] * dC[m*N + n];
            }
            dB[k*N + n] = sum;
        }
    }
}
// Compute dW = X^T @ dY   (In x Out)  ; X (B x In), dY (B x Out)
void linear_dW_impl_optimized(const float* X, const float* dY, float* dW,
                              int B, int In, int Out) {
    assert(X && dY && dW);
    if (B <= 0 || In <= 0 || Out <= 0) return;

    const int TILE_I = 32;   // tile over In (rows of dW)
    const int TILE_O = 64;   // tile over Out (cols of dW)
    const int TILE_B = 64;   // accumulate blocks of batch
    const int VEC = 8;
    
    // Zero dW
    std::fill(dW, dW + (size_t)In * Out, 0.0f);

    // For each block of (i0..i1) x (o0..o1) compute partial sums
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i0 = 0; i0 < In; i0 += TILE_I) {
        for (int o0 = 0; o0 < Out; o0 += TILE_O) {
            int i1 = std::min(i0 + TILE_I, In);
            int o1 = std::min(o0 + TILE_O, Out);

            // Use a local buffer to accumulate over B blocks to reduce contention
            std::vector<float> local((size_t)(i1 - i0) * (o1 - o0));
            std::fill(local.begin(), local.end(), 0.0f);

            for (int b0 = 0; b0 < B; b0 += TILE_B) {
                int b1 = std::min(b0 + TILE_B, B);

                for (int b = b0; b < b1; ++b) {
                    const float* Xrow = X + (size_t)b * In;
                    const float* DYrow = dY + (size_t)b * Out;

                    // accumulate into local[(i-i0)*(o1-o0) + (o-o0)]
                    for (int i = i0; i < i1; ++i) {
                        __m256 x_bcast = _mm256_set1_ps(Xrow[i]);
                        int o = o0;
                        // vector loop
                        for (; o + VEC <= o1; o += VEC) {
                            __m256 dy_vec = _mm256_loadu_ps(DYrow + o);
                            __m256 acc = _mm256_loadu_ps(&local[(i - i0) * (o1 - o0) + (o - o0)]);
                            acc = _mm256_fmadd_ps(x_bcast, dy_vec, acc);
                            _mm256_storeu_ps(&local[(i - i0) * (o1 - o0) + (o - o0)], acc);
                        }
                        for (; o < o1; ++o) {
                            local[(i - i0) * (o1 - o0) + (o - o0)] += Xrow[i] * DYrow[o];
                        }
                    }
                } // b block
            } // b0

            // write local into global dW
            for (int i = i0; i < i1; ++i) {
                float* dWrow = dW + (size_t)i * Out;
                int o = o0;
                // vector copy
                for (; o + VEC <= o1; o += VEC) {
                    __m256 v = _mm256_loadu_ps(&local[(i - i0) * (o1 - o0) + (o - o0)]);
                    __m256 orig = _mm256_loadu_ps(dWrow + o);
                    orig = _mm256_add_ps(orig, v);
                    _mm256_storeu_ps(dWrow + o, orig);
                }
                for (; o < o1; ++o) dWrow[o] += local[(i - i0) * (o1 - o0) + (o - o0)];
            }
        }
    }
}

// Compute dX = dY @ W^T   (B x In) ; dY (B x Out), W (In x Out)
void linear_dX_impl_optimized(const float* dY, const float* W, float* dX,
                              int B, int In, int Out) {
    assert(dY && W && dX);
    if (B <= 0 || In <= 0 || Out <= 0) return;

    const int TILE_B = 32;
    const int TILE_I = 32;
    const int TILE_O = 64;
    const int VEC = 8;

    // Zero dX
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B; ++b) {
        float* dxrow = dX + (size_t)b * In;
        for (int i = 0; i < In; ++i) dxrow[i] = 0.0f;
    }

    // We compute for each b and i: dX[b,i] += sum_o dY[b,o] * W[i,o]
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int b0 = 0; b0 < B; b0 += TILE_B) {
        for (int i0 = 0; i0 < In; i0 += TILE_I) {
            int b1 = std::min(b0 + TILE_B, B);
            int i1 = std::min(i0 + TILE_I, In);

            for (int o0 = 0; o0 < Out; o0 += TILE_O) {
                int o1 = std::min(o0 + TILE_O, Out);

                for (int b = b0; b < b1; ++b) {
                    const float* dyrow = dY + (size_t)b * Out;
                    float* dxrow = dX + (size_t)b * In;

                    for (int i = i0; i < i1; ++i) {
                        // compute dot(dyrow[o0..o1), W[i*Out + o0..o1))
                        __m256 acc = _mm256_setzero_ps();
                        int o = o0;
                        for (; o + VEC <= o1; o += VEC) {
                            __m256 dy_vec = _mm256_loadu_ps(dyrow + o);
                            __m256 w_vec  = _mm256_loadu_ps(W + (size_t)i * Out + o);
                            acc = _mm256_fmadd_ps(dy_vec, w_vec, acc);
                        }
                        float tmp[VEC];
                        _mm256_storeu_ps(tmp, acc);
                        float sum = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
                        for (; o < o1; ++o) sum += dyrow[o] * W[i * Out + o];
                        dxrow[i] += sum;
                    }
                } // b
            } // o0
        } // i0
    } // b0
}

// Compute db = sum_rows(dY)  (1 x Out)
void linear_db_impl_optimized(const float* dY, float* db, int B, int Out) {
    assert(dY && db);
    if (B <= 0 || Out <= 0) return;

    const int VEC = 8;
    // zero
    std::fill(db, db + Out, 0.0f);

    // Accumulate in parallel with per-thread local buffer to avoid atomic adds
    int num_threads = omp_get_max_threads();
    std::vector<std::vector<float>> local(num_threads, std::vector<float>(Out, 0.0f));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto &loc = local[tid];

        #pragma omp for schedule(static)
        for (int b = 0; b < B; ++b) {
            const float* dyrow = dY + (size_t)b * Out;
            int o = 0;
            for (; o + VEC <= Out; o += VEC) {
                __m256 v = _mm256_loadu_ps(dyrow + o);
                __m256 cur = _mm256_loadu_ps(&loc[o]);
                cur = _mm256_add_ps(cur, v);
                _mm256_storeu_ps(&loc[o], cur);
            }
            for (; o < Out; ++o) loc[o] += dyrow[o];
        } // for b
    } // parallel

    // reduce locals into db
    for (int t = 0; t < num_threads; ++t) {
        for (int o = 0; o < Out; ++o) db[o] += local[t][o];
    }
}
} // extern "C"

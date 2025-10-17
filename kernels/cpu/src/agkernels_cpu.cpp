// // kernels/cpu/src/agkernels_cpu.cpp
// #include "ad/kernels_api.hpp"
// #include <cstdint>

// extern "C" {

// // ---------------- reference implementations ----------------
//  void relu_impl(const float* x, float* y, int64_t n){
//   for (int64_t i=0;i<n;++i) y[i] = x[i] > 0.0f ? x[i] : 0.0f;
// }

//  void matmul_impl(const float* A, const float* B, float* C,
//                         int M, int K, int N){
//   // C(MxN) = A(MxK) * B(KxN)
//   for (int i=0;i<M;++i){
//     for (int j=0;j<N;++j){
//       float acc = 0.f;
//       const float* Ai = A + i*K;
//       for (int k=0;k<K;++k) acc += Ai[k] * B[k*N + j];  
//       C[i*N + j] = acc;
//     }
//   }
// }

// // ---------------- required export ----------------
// AG_EXPORT int ag_get_cpu_kernels_v1(struct ag_cpu_v1* out){
//   if (!out) return -1;
//   out->abi_version = AG_KERNELS_ABI_V1;
//   out->relu   = &relu_impl;
//   out->matmul = &matmul_impl;
//   return 0;
// }

// } // extern "C"
//===================================================================================================================
#include "ad/kernels_api.hpp"
#include <cstdint>
#include <cmath>
// Headers for CPU intrinsics (AVX/FMA) and OpenMP
#include <immintrin.h>
#include <omp.h>
#include "matker.cuh"


extern "C" {

// ---------------- Optimized Implementations ----------------

/**
 * Optimized ReLU using AVX2 and OpenMP.
 * Processes 8 floats at a time and parallelizes across all cores.
 */
void relu_impl_optimized(const float* x, float* y, int64_t n) {
    const __m256 zeros = _mm256_setzero_ps(); // A vector of 8 zeros

    #pragma omp parallel for
    for (int64_t i = 0; i < n; i += 8) {
        // Ensure we don't read past the end of the array
        if (i + 8 <= n) {
            __m256 x_vec = _mm256_loadu_ps(x + i);      // Load 8 floats from x
            __m256 y_vec = _mm256_max_ps(x_vec, zeros); // Compute max(x, 0) for all 8 floats
            _mm256_storeu_ps(y + i, y_vec);            // Store 8 results back to y
        } else {
            // Handle the remaining elements one by one
            for (int64_t j = i; j < n; ++j) {
                y[j] = x[j] > 0.0f ? x[j] : 0.0f;
            }
        }
    }
}

void leakyrelu_impl_optimized(const float* x, float* y, int64_t n, float alpha) {
    const __m256 kZero  = _mm256_setzero_ps();
    const __m256 kAlpha = _mm256_set1_ps(alpha);

    #pragma omp parallel for
    for (int64_t i = 0; i < n; i += 8) {
        if (i + 8 <= n) {
            // Load 8 float values
            __m256 x_vec = _mm256_loadu_ps(x + i);

            // Compare x > 0 → mask (true = 0xFFFFFFFF)
            __m256 mask = _mm256_cmp_ps(x_vec, kZero, _CMP_GT_OS);

            // Compute alpha * x for negative elements
            __m256 neg_part = _mm256_mul_ps(x_vec, kAlpha);

            // Blend positive and negative parts: y = (mask ? x : alpha*x)
            __m256 y_vec = _mm256_blendv_ps(neg_part, x_vec, mask);

            // Store results
            _mm256_storeu_ps(y + i, y_vec);
        } else {
            // Tail elements
            for (int64_t j = i; j < n; ++j) {
                float v = x[j];
                y[j] = v > 0.0f ? v : alpha * v;
            }
        }
    }
}

void gemm_impl_optimized(const float* A, const float* B,  float* C, int M, int K, int N) {
    int q = N;
    int p = K;
    int s = p+q;
    if(s)
    {
    run_cuda_gemm(A, B, C, M);
    }
}


void matmul_impl_naive(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
}
// Optimized GELU using AVX2 + OpenMP
void gelu_impl_optimized(const float* x, float* y, int64_t n) {
    const __m256 k0_5      = _mm256_set1_ps(0.5f);
    const __m256 k0_044715 = _mm256_set1_ps(0.044715f);
    const __m256 kSqrt2OverPi = _mm256_set1_ps(0.7978845608028654f); // sqrt(2/pi)

    #pragma omp parallel for
    for (int64_t i = 0; i < n; i += 8) {
        if (i + 8 <= n) {
            // Load 8 float values
            __m256 x_vec = _mm256_loadu_ps(x + i);

            // x^3
            __m256 x2 = _mm256_mul_ps(x_vec, x_vec);
            __m256 x3 = _mm256_mul_ps(x2, x_vec);

            // u = sqrt(2/pi) * (x + 0.044715 * x^3)
            __m256 term = _mm256_fmadd_ps(k0_044715, x3, x_vec);
            __m256 u = _mm256_mul_ps(term, kSqrt2OverPi);

            // tanh(u) (approximation)
            // AVX2 has no native tanh, so use polynomial approximation.
            // tanh(u) ≈ u * (27 + u^2) / (27 + 9u^2)
            __m256 u2 = _mm256_mul_ps(u, u);
            __m256 num = _mm256_fmadd_ps(u2, _mm256_set1_ps(1.0f), _mm256_set1_ps(27.0f));
            __m256 den = _mm256_fmadd_ps(u2, _mm256_set1_ps(9.0f), _mm256_set1_ps(27.0f));
            __m256 th = _mm256_div_ps(_mm256_mul_ps(u, num), den);

            // 0.5 * x * (1 + tanh(u))
            __m256 one_plus_th = _mm256_add_ps(th, _mm256_set1_ps(1.0f));
            __m256 result = _mm256_mul_ps(k0_5, _mm256_mul_ps(x_vec, one_plus_th));

            _mm256_storeu_ps(y + i, result);
        } else {
            // tail elements
            for (int64_t j = i; j < n; ++j) {
                float v = x[j];
                float u = 0.7978845608028654f * (v + 0.044715f * v * v * v);
                float th = std::tanh(u);
                y[j] = 0.5f * v * (1.0f + th);
            }
        }
    }
}

/**
 * Optimized MatMul using AVX2, FMA, OpenMP, and cache blocking.
 * C(MxN) = A(MxK) * B(KxN)
 */
void matmul_impl_optimized(const float* A, const float* B, float* C, int M, int K, int N) {
    // Simple blocked implementation with safe bounds and AVX2 inner loop.
    const int TILE_M = 32;   // tune on your CPU
    const int TILE_K = 32;
    const int TILE_N = 32;
    const int VEC = 8;

    // Zero C entirely (safe)
    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        float* crow = C + i * N;
        for (int j = 0; j < N; ++j) crow[j] = 0.0f;
    }

    #pragma omp parallel for collapse(2)
    for (int i0 = 0; i0 < M; i0 += TILE_M) {
        for (int j0 = 0; j0 < N; j0 += TILE_N) {
            int i1 = std::min(i0 + TILE_M, M);
            int j1 = std::min(j0 + TILE_N, N);

            for (int k0 = 0; k0 < K; k0 += TILE_K) {
                int k1 = std::min(k0 + TILE_K, K);

                for (int i = i0; i < i1; ++i) {
                    const float* Arow = A + i * K;
                    float* Crow = C + i * N;

                    for (int k = k0; k < k1; ++k) {
                        // broadcast A[i,k]
                        __m256 a_vec = _mm256_set1_ps(Arow[k]);

                        int j = j0;
                        // vectorized inner loop (process groups of 8)
                        for ( ; j + VEC <= j1; j += VEC) {
                            __m256 b_vec = _mm256_loadu_ps(B + k * N + j);   // B[k, j..]
                            __m256 c_vec = _mm256_loadu_ps(Crow + j);       // C[i, j..]
                            c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                            _mm256_storeu_ps(Crow + j, c_vec);
                        }
                        // tail scalar
                        for ( ; j < j1; ++j) {
                            Crow[j] += Arow[k] * B[k * N + j];
                        }
                    }
                } // i
            } // k0
        } // j0
    } // i0
}


void matmul_impl_cudatile(const float* A, const float* B, float* C, int M, int K, int N) {
    // This is a placeholder for a CUDA-tiled implementation.
    // In a real scenario, this function would offload computation to a GPU.
    // For now, we will just call the naive implementation as a stub.
    int q = N;
    int p = K;
    int s = p+q;
    if(s)
    {
    run_cuda_matrix(A, B, C, M);
    }
}
static inline __m256 log256_approx(__m256 x) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 ln2 = _mm256_set1_ps(0.6931471805599453f);
    const __m256i exp_mask = _mm256_set1_epi32(0x7F800000);
    const __m256i mant_mask = _mm256_set1_epi32(0x007FFFFF);
    const __m256i bias = _mm256_set1_epi32(127);
    __m256i xi = _mm256_castps_si256(x);
    __m256i exp_i = _mm256_srli_epi32(_mm256_and_si256(xi, exp_mask), 23);
    __m256i mant_i = _mm256_and_si256(xi, mant_mask);
    __m256 m = _mm256_add_ps(one, _mm256_mul_ps(_mm256_cvtepi32_ps(mant_i), _mm256_set1_ps(1.0f / (1 << 23))));
    __m256 e = _mm256_sub_ps(_mm256_cvtepi32_ps(exp_i), _mm256_cvtepi32_ps(bias));
    __m256 t = _mm256_sub_ps(m, one);
    __m256 c1 = _mm256_set1_ps(1.0f);
    __m256 c2 = _mm256_set1_ps(-0.5f);
    __m256 c3 = _mm256_set1_ps(0.3333f);
    __m256 c4 = _mm256_set1_ps(-0.25f);
    __m256 c5 = _mm256_set1_ps(0.2f);
    __m256 p = c5;
    p = _mm256_fmadd_ps(p, t, c4);
    p = _mm256_fmadd_ps(p, t, c3);
    p = _mm256_fmadd_ps(p, t, c2);
    p = _mm256_fmadd_ps(p, t, c1);
    __m256 logm = _mm256_mul_ps(p, t);
    return _mm256_fmadd_ps(e, ln2, logm);
}
static inline __m256 exp256_approx(__m256 x) {
    // Constants
    const __m256 ln2_inv = _mm256_set1_ps(1.4426950408889634f); // 1/ln(2)
    const __m256 ln2    = _mm256_set1_ps(0.6931471805599453f); // ln(2)

    // polynomial coeffs for e^f on f in [-0.5,0.5] (minimax / estrin-friendly)
    // Coeffs from a common 6th-degree approximation (order/accuracy tradeoff)
    const __m256 c0 = _mm256_set1_ps(1.0f);
    const __m256 c1 = _mm256_set1_ps(1.0f);
    const __m256 c2 = _mm256_set1_ps(0.4999999975f);   // 1/2
    const __m256 c3 = _mm256_set1_ps(0.166666648f);    // 1/6
    const __m256 c4 = _mm256_set1_ps(0.041666663f);    // 1/24
    const __m256 c5 = _mm256_set1_ps(0.0083333315f);   // 1/120
    const __m256 c6 = _mm256_set1_ps(0.0013888889f);   // 1/720

    // clamp x to avoid overflow/underflow
    const __m256 max_x = _mm256_set1_ps(88.0f);   // exp(88) ~ 1e38
    const __m256 min_x = _mm256_set1_ps(-88.0f);
    x = _mm256_min_ps(x, max_x);
    x = _mm256_max_ps(x, min_x);

    // compute n = floor(x / ln2 + 0.5)
    __m256 fx = _mm256_mul_ps(x, ln2_inv);
    // round to nearest int (using floor(x+0.5) trick)
    __m256 fx_round = _mm256_round_ps(fx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    // f = x - n * ln2
    __m256 t = _mm256_mul_ps(fx_round, ln2);
    __m256 f = _mm256_sub_ps(x, t);

    // polynomial: e^f ≈ c0 + f*(c1 + f*(c2 + f*(...)))
    // Use Horner
    __m256 f2 = _mm256_mul_ps(f, f);

    __m256 p = c6;
    p = _mm256_fmadd_ps(p, f, c5);
    p = _mm256_fmadd_ps(p, f, c4);
    p = _mm256_fmadd_ps(p, f, c3);
    p = _mm256_fmadd_ps(p, f, c2);
    p = _mm256_fmadd_ps(p, f, c1);
    p = _mm256_fmadd_ps(p, f, c0); // p now holds polynomial result

    // reconstruct exp(x) = 2^n * p
    // compute 2^n by building float bits: exp2(n) = reinterpret_cast<float>( (n + 127) << 23 )
    // need to convert fx_round (float vector) to integer vector
    __m256i n_int = _mm256_cvtps_epi32(fx_round); // rounds to nearest int
    // add exponent bias (127)
    n_int = _mm256_add_epi32(n_int, _mm256_set1_epi32(127));
    // shift left 23 to place in exponent bits
    n_int = _mm256_slli_epi32(n_int, 23);
    // reinterpret as float
    __m256 pow2n = _mm256_castsi256_ps(n_int);

    // result = p * 2^n
    __m256 result = _mm256_mul_ps(p, pow2n);
    return result;
}

// --------------------------------------------
// sigmoid kernel (AVX2 + OpenMP)
// --------------------------------------------
void sigmoid_impl_optimized(const float* x, float* y, int64_t n) {
    const __m256 one = _mm256_set1_ps(1.0f);

    #pragma omp parallel for
    for (int64_t i = 0; i < n; i += 8) {
        if (i + 8 <= n) {
            // load 8 values
            __m256 xv = _mm256_loadu_ps(x + i);

            // compute -x
            __m256 negx = _mm256_sub_ps(_mm256_setzero_ps(), xv);

            // approximate exp(-x)
            __m256 e = exp256_approx(negx);

            // denom = 1 + e
            __m256 denom = _mm256_add_ps(one, e);

            // sigmoid = 1 / denom
            __m256 sig = _mm256_div_ps(one, denom);

            // store
            _mm256_storeu_ps(y + i, sig);
        } else {
            // tail scalar fallback
            for (int64_t j = i; j < n; ++j) {
                float v = x[j];
                y[j] = 1.0f / (1.0f + std::exp(-v));
            }
        }
    }
}
void tanh_impl_optimized(const float* x, float* y, int64_t n) {
    const __m256 one        = _mm256_set1_ps(1.0f);
    const __m256 neg_one    = _mm256_set1_ps(-1.0f);
    const __m256 c27        = _mm256_set1_ps(27.0f);
    const __m256 c9         = _mm256_set1_ps(9.0f);
    const __m256 clamp_abs  = _mm256_set1_ps(10.0f); // beyond this, tanh ~ ±1
    const __m256 zero       = _mm256_setzero_ps();

    // mask for absolute value (0x7FFFFFFF)
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

    #pragma omp parallel for
    for (int64_t i = 0; i < n; i += 8) {
        if (i + 8 <= n) {
            // load 8 floats
            __m256 xv = _mm256_loadu_ps(x + i);

            // abs(x)
            __m256 ax = _mm256_and_ps(xv, abs_mask);

            // compute x^2
            __m256 x2 = _mm256_mul_ps(xv, xv);

            // numerator = x * (27 + x^2)
            __m256 num = _mm256_add_ps(c27, x2);        // (27 + x^2)
            __m256 num_x = _mm256_mul_ps(xv, num);      // x * (27 + x^2)

            // denominator = (27 + 9*x^2)
            __m256 den = _mm256_fmadd_ps(c9, x2, c27);  // 9*x^2 + 27

            // approx = num_x / den
            __m256 approx = _mm256_div_ps(num_x, den);

            // For large |x|, use sign(x) * 1.0 (tanh saturates)
            __m256 mask_large = _mm256_cmp_ps(ax, clamp_abs, _CMP_GT_OS); // true where abs(x) > 10

            // compute sign(x): 1.0f if x >= 0 else -1.0f
            __m256 sign_mask = _mm256_cmp_ps(xv, zero, _CMP_GE_OS);
            __m256 sign_val  = _mm256_blendv_ps(neg_one, one, sign_mask);

            // select: if mask_large then sign_val else approx
            __m256 res = _mm256_blendv_ps(approx, sign_val, mask_large);

            // store result
            _mm256_storeu_ps(y + i, res);
        } else {
            // tail scalar fallback
            for (int64_t j = i; j < n; ++j) {
                float v = x[j];
                float av = std::fabs(v);
                if (av > 10.0f) {
                    y[j] = (v >= 0.0f) ? 1.0f : -1.0f;
                } else {
                    float x2s = v * v;
                    float num = v * (27.0f + x2s);
                    float den = 27.0f + 9.0f * x2s;
                    y[j] = num / den;
                }
            }
        }
    }
}

// Softplus optimized kernel: softplus(x) = log(1 + exp(x))
void softplus_impl_optimized(const float* x, float* y, int64_t n) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

    // Coefficients for log1p truncated series:
    // log1p(z) ≈ z - z^2/2 + z^3/3 - z^4/4 + z^5/5 - z^6/6
    const __m256 L6 = _mm256_set1_ps(-1.0f / 6.0f);
    const __m256 L5 = _mm256_set1_ps( 1.0f / 5.0f);
    const __m256 L4 = _mm256_set1_ps(-1.0f / 4.0f);
    const __m256 L3 = _mm256_set1_ps( 1.0f / 3.0f);
    const __m256 L2 = _mm256_set1_ps(-1.0f / 2.0f);
    const __m256 L1 = _mm256_set1_ps( 1.0f );

    #pragma omp parallel for
    for (int64_t i = 0; i < n; i += 8) {
        if (i + 8 <= n) {
            // load 8 floats
            __m256 xv = _mm256_loadu_ps(x + i);

            // abs and sign
            __m256 ax = _mm256_and_ps(xv, abs_mask); // abs(x)
            // compute -abs(x) for exp
            __m256 neg_ax = _mm256_sub_ps(zero, ax);

            // z = exp(-|x|)  (z in (0,1])
            __m256 z = exp256_approx(neg_ax);

            // compute log1p(z) via truncated alternating series
            // p = (((((L6*z + L5)*z + L4)*z + L3)*z + L2)*z + L1) * z
            __m256 p = L6;
            p = _mm256_fmadd_ps(p, z, L5);
            p = _mm256_fmadd_ps(p, z, L4);
            p = _mm256_fmadd_ps(p, z, L3);
            p = _mm256_fmadd_ps(p, z, L2);
            p = _mm256_fmadd_ps(p, z, L1);
            __m256 log1p_z = _mm256_mul_ps(p, z);

            // For x > 0 : softplus = x + log1p(exp(-x)) = x + log1p_z
            // For x <=0 : softplus = log1p(exp(x)) = log1p_z
            __m256 mask_pos = _mm256_cmp_ps(xv, zero, _CMP_GT_OS); // true where x > 0
            __m256 add_part = _mm256_add_ps(xv, log1p_z);          // x + log1p_z
            __m256 res = _mm256_blendv_ps(log1p_z, add_part, mask_pos);

            // store result
            _mm256_storeu_ps(y + i, res);
        } else {
            // tail scalar fallback with full precision
            for (int64_t j = i; j < n; ++j) {
                float v = x[j];
                if (v > 0.0f) {
                    float z = std::exp(-v);
                    y[j] = v + std::log1p(z);
                } else {
                    float z = std::exp(v);
                    y[j] = std::log1p(z);
                }
            }
        }
    }
}

void exp_impl_optimized(const float* x, float* y, int64_t n) {
    // constants
    const __m256 ln2_inv = _mm256_set1_ps(1.4426950408889634f); // 1/ln(2)
    const __m256 ln2     = _mm256_set1_ps(0.6931471805599453f);
    const __m256 max_x   = _mm256_set1_ps(88.0f);
    const __m256 min_x   = _mm256_set1_ps(-88.0f);

    // polynomial coeffs for e^f on f in [-0.5,0.5] (Horner order)
    const __m256 c6 = _mm256_set1_ps(0.0013888889f);
    const __m256 c5 = _mm256_set1_ps(0.0083333315f);
    const __m256 c4 = _mm256_set1_ps(0.041666663f);
    const __m256 c3 = _mm256_set1_ps(0.166666648f);
    const __m256 c2 = _mm256_set1_ps(0.4999999975f);
    const __m256 c1 = _mm256_set1_ps(1.0f);
    const __m256 c0 = _mm256_set1_ps(1.0f);

    #pragma omp parallel for
    for (int64_t i = 0; i < n; i += 8) {
        if (i + 8 <= n) {
            __m256 xv = _mm256_loadu_ps(x + i);

            // clamp
            xv = _mm256_min_ps(xv, max_x);
            xv = _mm256_max_ps(xv, min_x);

            // fx = x / ln2
            __m256 fx = _mm256_mul_ps(xv, ln2_inv);

            // n = round(fx)
            __m256 fx_round = _mm256_round_ps(fx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

            // f = x - n * ln2
            __m256 t = _mm256_mul_ps(fx_round, ln2);
            __m256 f = _mm256_sub_ps(xv, t);

            // polynomial e^f via Horner
            __m256 p = c6;
            p = _mm256_fmadd_ps(p, f, c5);
            p = _mm256_fmadd_ps(p, f, c4);
            p = _mm256_fmadd_ps(p, f, c3);
            p = _mm256_fmadd_ps(p, f, c2);
            p = _mm256_fmadd_ps(p, f, c1);
            p = _mm256_fmadd_ps(p, f, c0); // p approx e^f

            // reconstruct 2^n via exponent bits
            __m256i n_int = _mm256_cvtps_epi32(fx_round);       // fx_round -> int
            n_int = _mm256_add_epi32(n_int, _mm256_set1_epi32(127));
            n_int = _mm256_slli_epi32(n_int, 23);
            __m256 pow2n = _mm256_castsi256_ps(n_int);

            // result = p * 2^n
            __m256 res = _mm256_mul_ps(p, pow2n);
            _mm256_storeu_ps(y + i, res);
        } else {
            // scalar tail
            for (int64_t j = i; j < n; ++j) y[j] = std::exp(x[j]);
        }
    }
}
void log_impl_optimized(const float* x, float* y, int64_t n) {
    // constants
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256i exponent_mask = _mm256_set1_epi32(0x7F800000);
    const __m256i mantissa_mask = _mm256_set1_epi32(0x007FFFFF);
    const __m256i bias = _mm256_set1_epi32(127);

    // polynomial coeffs for log on normalized mantissa in [1,2)
    // Use a minimax-friendly polynomial (order 5)
    const __m256 L1 = _mm256_set1_ps( 0.9999998650f);
    const __m256 L2 = _mm256_set1_ps(-0.4999998807f);
    const __m256 L3 = _mm256_set1_ps( 0.3333318000f);
    const __m256 L4 = _mm256_set1_ps(-0.2499993000f);
    const __m256 L5 = _mm256_set1_ps( 0.1999990000f);
    const __m256 ln2 = _mm256_set1_ps(0.6931471805599453f);

    #pragma omp parallel for
    for (int64_t i = 0; i < n; i += 8) {
        if (i + 8 <= n) {
            __m256 xv = _mm256_loadu_ps(x + i);

            // handle small/invalid inputs by scalar fallback
            // create mask for xv <= 0
            __m256 le_mask = _mm256_cmp_ps(xv, _mm256_set1_ps(0.0f), _CMP_LE_OQ);
            if (_mm256_movemask_ps(le_mask) != 0) {
                // some non-positive elements - fallback elementwise
                for (int64_t j = i; j < i + 8; ++j) {
                    y[j] = (x[j] > 0.0f) ? std::log(x[j]) : -INFINITY;
                }
                continue;
            }

            // reinterpret as int to get exponent and mantissa
            __m256i xi = _mm256_castps_si256(xv);
            __m256i exp_i = _mm256_srli_epi32(_mm256_and_si256(xi, exponent_mask), 23);
            __m256i mant_i = _mm256_and_si256(xi, mantissa_mask);

            // normalized mantissa m = 1 + mantissa / 2^23
            __m256 mant_f = _mm256_cvtepi32_ps(mant_i);
            const float inv_2_23 = 1.0f / float(1 << 23);
            __m256 m = _mm256_add_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(mant_f, _mm256_set1_ps(inv_2_23)));

            // exponent as float: e = exp_i - bias
            __m256 e = _mm256_sub_ps(_mm256_cvtepi32_ps(exp_i), _mm256_cvtepi32_ps(bias));

            // compute r = (m - 1) / (m + 1) for improved convergence (optional)
            // here we use simple polynomial on (m - 1), with variable t = m - 1 in [0,1)
            __m256 t = _mm256_sub_ps(m, _mm256_set1_ps(1.0f));
            // compute polynomial log(1+t) ~ t*(L1 + t*(L2 + t*(L3 + t*(L4 + t*L5))))
            __m256 p = L5;
            p = _mm256_fmadd_ps(p, t, L4);
            p = _mm256_fmadd_ps(p, t, L3);
            p = _mm256_fmadd_ps(p, t, L2);
            p = _mm256_fmadd_ps(p, t, L1);
            __m256 logm = _mm256_mul_ps(p, t);

            // full log(x) = e*ln2 + log(m)
            __m256 res = _mm256_fmadd_ps(e, ln2, logm);
            _mm256_storeu_ps(y + i, res);
        } else {
            // scalar tail
            for (int64_t j = i; j < n; ++j) {
                y[j] = (x[j] > 0.0f) ? std::log(x[j]) : -INFINITY;
            }
        }
    }
}
void sqrt_impl_optimized(const float* x, float* y, int64_t n) {
    const __m256 zero = _mm256_set1_ps(0.0f);

    #pragma omp parallel for
    for (int64_t i = 0; i < n; i += 8) {
        if (i + 8 <= n) {
            // load 8 floats
            __m256 xv = _mm256_loadu_ps(x + i);
            // clamp negatives to zero (to avoid NaN)
            __m256 mask = _mm256_cmp_ps(xv, zero, _CMP_LT_OS);
            xv = _mm256_blendv_ps(xv, zero, mask);
            // compute sqrt
            __m256 yv = _mm256_sqrt_ps(xv);
            _mm256_storeu_ps(y + i, yv);
        } else {
            // scalar tail
            for (int64_t j = i; j < n; ++j)
                y[j] = x[j] > 0.0f ? std::sqrt(x[j]) : 0.0f;
        }
    }
}
void pow_impl_optimized(const float* x, float* y, int64_t n, float exponent) {
    const __m256 expv = _mm256_set1_ps(exponent);
    const __m256 zero = _mm256_set1_ps(0.0f);
    const __m256 min_val = _mm256_set1_ps(1e-20f); // to prevent log(0)

    #pragma omp parallel for
    for (int64_t i = 0; i < n; i += 8) {
        if (i + 8 <= n) {
            // load x
            __m256 xv = _mm256_loadu_ps(x + i);
            // clamp to avoid log(0)
            xv = _mm256_max_ps(xv, min_val);

            // compute log(x)
            __m256 lv = log256_approx(xv);

            // exponentiate: exp(exponent * log(x))
            __m256 pv = _mm256_mul_ps(lv, expv);
            __m256 yv = exp256_approx(pv);

            _mm256_storeu_ps(y + i, yv);
        } else {
            // scalar fallback
            for (int64_t j = i; j < n; ++j)
                y[j] = std::pow(x[j], exponent);
        }
    }
}
// ---------------- required export ----------------
// This part exports the new optimized functions.
AG_EXPORT int ag_get_cpu_kernels_v1(struct ag_cpu_v1* out){
  if (!out) return -1;
    out->abi_version = AG_KERNELS_ABI_V1;
    out->relu   = &relu_impl_optimized;
    out->matmul = &matmul_impl_optimized;
    out->fmab = &gemm_impl_optimized;
    out->gelu = &gelu_impl_optimized;
    out->leakyrelu = &leakyrelu_impl_optimized;
    out->sigmoid = &sigmoid_impl_optimized;
    out->tanh = &tanh_impl_optimized;
    out->softmax = &softplus_impl_optimized;
    out->exp = &exp_impl_optimized;
    out->log = &log_impl_optimized; 
    out->sqrt = &sqrt_impl_optimized;
    out->pow = &pow_impl_optimized;  
  //backwards
    out->relu_bwd = &relu_bwd_impl_optimized;
    out->leakyrelu_bwd = &leakyrelu_bwd_impl_optimized;
    out->sigmoid_bwd_from_s = &sigmoid_bwd_impl_optimized_from_s;
    out->tanh_bwd_from_t = &tanh_bwd_impl_optimized_from_t;
    out->gelu_bwd = &gelu_bwd_impl_optimized;
    out->softplus_bwd = &softplus_bwd_impl_optimized;
    out->exp_bwd_from_y = &exp_bwd_impl_optimized_from_y;
    out->log_bwd = &log_bwd_impl_optimized;
    out->sqrt_bwd_from_y = &sqrt_bwd_impl_optimized_from_y;
    out->matmul_bwd_dA = &matmul_bwd_dA_impl_optimized;
    out->matmul_bwd_dB = &matmul_bwd_dB_impl_optimized;
  return 0;
}

} // extern "C"
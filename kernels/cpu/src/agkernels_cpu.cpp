// =============================================
// kernels/cpu/src/agkernels_cpu.cpp
// =============================================

#include "ad/kernels_api.hpp"
#include <cstdint>
#include <cmath>
// Headers for CPU intrinsics (AVX/FMA) and OpenMP
#include <immintrin.h>
#include <omp.h>
#include <cstdio>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cassert>
#include <vector>
#include <cstring>
#include <algorithm>
// #include "adkernels_cpu.cpp"
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
    // __m256 f2 = _mm256_mul_ps(f, f);

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
    // const __m256 one = _mm256_set1_ps(1.0f);
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
    // const __m256 one = _mm256_set1_ps(1.0f);
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
    // const __m256 zero = _mm256_set1_ps(0.0f);
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
void linear_impl_optimized(const float* X, const float* W, const float* b, float* Y,
                           int B, int In, int Out) {
    assert(X != nullptr && W != nullptr && Y != nullptr);
    if (B <= 0 || In <= 0 || Out <= 0) return;

    // Tunable tile sizes (good starting points)
    constexpr int TILE_B = 16;  // tile over batch
    constexpr int TILE_O = 64;  // tile over output (columns)
    constexpr int TILE_I = 64;  // tile over input (k)

    const int VEC = 8; // AVX2 vector width

    // Initialize output with bias if provided, otherwise zero
    #pragma omp parallel for schedule(static)
    for (int bi = 0; bi < B; ++bi) {
        float* Yrow = Y + (size_t)bi * Out;
        if (b) {
            for (int j = 0; j < Out; ++j) Yrow[j] = b[j];
        } else {
            for (int j = 0; j < Out; ++j) Yrow[j] = 0.0f;
        }
    }

    // Blocked matmul-like accumulation: for each block of batch x out, accumulate over In
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int b0 = 0; b0 < B; b0 += TILE_B) {
        for (int o0 = 0; o0 < Out; o0 += TILE_O) {
            int b1 = std::min(b0 + TILE_B, B);
            int o1 = std::min(o0 + TILE_O, Out);

            for (int k0 = 0; k0 < In; k0 += TILE_I) {
                int k1 = std::min(k0 + TILE_I, In);

                // Process the tile (b0..b1) x (o0..o1), accumulating over k0..k1
                for (int bi = b0; bi < b1; ++bi) {
                    const float* Xrow = X + (size_t)bi * In;
                    float* Yrow = Y + (size_t)bi * Out;

                    for (int k = k0; k < k1; ++k) {
                        // broadcast X[bi,k]
                        __m256 a_vec = _mm256_set1_ps(Xrow[k]);
                        const float* Wrow = W + (size_t)k * Out; // W[k, 0..Out-1]

                        int oj = o0;
                        // vectorized loop over outputs
                        for (; oj + VEC <= o1; oj += VEC) {
                            __m256 w_vec = _mm256_loadu_ps(Wrow + oj);
                            __m256 y_vec = _mm256_loadu_ps(Yrow + oj);
                            y_vec = _mm256_fmadd_ps(a_vec, w_vec, y_vec); // y += a * w
                            _mm256_storeu_ps(Yrow + oj, y_vec);
                        }
                        // scalar tail
                        for (; oj < o1; ++oj) {
                            Yrow[oj] += Xrow[k] * Wrow[oj];
                        }
                    } // k
                } // bi
            } // k0
        } // o0
    } // b0
}



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
    // const __m256 one = _mm256_set1_ps(1.0f);
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
    // This function needs to compute dA(M,K) = dC(M,N) @ B^T(N,K).
    
    // 1. Allocate temporary memory for B transposed (B_t will have shape N,K)
    std::vector<float> B_t_vec( (size_t)N * K );
    float* B_t = B_t_vec.data();

    // 2. Transpose B into B_t. B is (K,N), B_t is (N,K).
    // This is a simple, parallelized transpose loop.
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            B_t[j * K + i] = B[i * N + j];
        }
    }

    // 3. Now we can call the existing optimized matmul.
    // The shapes are: dA(M,K) = dC(M,N) @ B_t(N,K)
    matmul_impl_optimized(dC, B_t, dA, M, N, K);
    
    // The temporary memory for B_t is automatically freed when B_t_vec goes out of scope.
}

// Compute dB = A^T @ dC
// A: [M,K], dC: [M,N] -> A^T: [K,M] @ [M,N] = [K,N]
void matmul_bwd_dB_impl_optimized(const float* A, const float* dC, float* dB, int M, int K, int N) {
    // This function needs to compute dB(K,N) = A^T(K,M) @ dC(M,N).

    // 1. Allocate temporary memory for A transposed (A_t will have shape K,M)
    std::vector<float> A_t_vec( (size_t)K * M );
    float* A_t = A_t_vec.data();

    // 2. Transpose A into A_t. A is (M,K), A_t is (K,M).
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            A_t[j * M + i] = A[i * K + j];
        }
    }

    // 3. Now we can call the existing optimized matmul.
    // The shapes are: dB(K,N) = A_t(K,M) @ dC(M,N)
    matmul_impl_optimized(A_t, dC, dB, K, M, N);

    // The temporary memory for A_t is automatically freed.
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
// ---------------- required export ----------------
// This part exports the new optimized functions.
AG_EXPORT int ag_get_cpu_kernels_v1(struct ag_cpu_v1* out){
  if (!out) return -1;
    out->abi_version = AG_KERNELS_ABI_V1;
    out->relu   = &relu_impl_optimized;
    out->matmul = &matmul_impl_optimized;
    // out->fmab = &gemm_impl_optimized;
    out->gelu = &gelu_impl_optimized;
    out->leakyrelu = &leakyrelu_impl_optimized;
    out->sigmoid = &sigmoid_impl_optimized;
    out->tanh = &tanh_impl_optimized;
    out->softplus = &softplus_impl_optimized;
    out->exp = &exp_impl_optimized;
    out->log = &log_impl_optimized; 
    out->sqrt = &sqrt_impl_optimized;
    out->pow = &pow_impl_optimized;  
    out->linear = &linear_impl_optimized;
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
    out->linear_dW = &linear_dW_impl_optimized;
    out->linear_dX = &linear_dX_impl_optimized;
    out->linear_db = &linear_db_impl_optimized;
  return 0;
}

} // extern "C"
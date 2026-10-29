#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

EXPORT void synaptic_column_avx2_forward(
    const float* __restrict__ x,
    const float* __restrict__ W_centroids,
    const float* __restrict__ b_resting,
    float* __restrict__ out_potentials,
    int D,
    int num_columns,
    float v_threshold
) {
    for (int c = 0; c < num_columns; ++c) {
        const float* W_row = W_centroids + (size_t)c * D;
        __m256 v_acc0 = _mm256_setzero_ps();
        __m256 v_acc1 = _mm256_setzero_ps();
        
        int j = 0;
        int unroll_limit = D - 16;
        for (; j <= unroll_limit; j += 16) {
            __m256 vx0 = _mm256_loadu_ps(x + j);
            __m256 vw0 = _mm256_loadu_ps(W_row + j);
            v_acc0 = _mm256_fmadd_ps(vx0, vw0, v_acc0);

            __m256 vx1 = _mm256_loadu_ps(x + j + 8);
            __m256 vw1 = _mm256_loadu_ps(W_row + j + 8);
            v_acc1 = _mm256_fmadd_ps(vx1, vw1, v_acc1);
        }

        v_acc0 = _mm256_add_ps(v_acc0, v_acc1);

        for (; j <= D - 8; j += 8) {
            __m256 vx = _mm256_loadu_ps(x + j);
            __m256 vw = _mm256_loadu_ps(W_row + j);
            v_acc0 = _mm256_fmadd_ps(vx, vw, v_acc0);
        }

        // Horizontal sum of v_acc0
        __m128 v_hi = _mm256_extractf128_ps(v_acc0, 1);
        __m128 v_lo = _mm256_castps256_ps128(v_acc0);
        __m128 v_sum = _mm_add_ps(v_lo, v_hi);
        v_sum = _mm_hadd_ps(v_sum, v_sum);
        v_sum = _mm_hadd_ps(v_sum, v_sum);
        float sum = _mm_cvtss_f32(v_sum);

        for (; j < D; ++j) {
            sum += x[j] * W_row[j];
        }

        sum += b_resting[c];
        out_potentials[c] = (sum >= v_threshold) ? sum : 0.0f;
    }
}

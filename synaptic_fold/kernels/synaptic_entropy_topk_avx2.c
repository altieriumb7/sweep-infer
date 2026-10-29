#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_K 16

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

// Fast Vectorized Top-K Selection and Shannon Entropy Calculation
EXPORT void synaptic_topk_entropy_avx2(
    const float* __restrict__ logits,
    int vocab_size,
    int k,
    int* __restrict__ out_indices,
    float* __restrict__ out_probs,
    float* __restrict__ out_entropy,
    float* __restrict__ out_margin
) {
    if (k > MAX_K) k = MAX_K;
    if (k <= 0) return;

    float top_vals[MAX_K];
    int top_idxs[MAX_K];

    // Initialize with minimum possible values
    for (int i = 0; i < k; ++i) {
        top_vals[i] = -1e30f;
        top_idxs[i] = -1;
    }
    float current_min = -1e30f;
    int min_slot = 0;

    // Vectorized scan across the entire 128,256 vocabulary
    int i = 0;
    int unroll_limit = vocab_size - 8;

    for (; i <= unroll_limit; i += 8) {
        __m256 v_logits = _mm256_loadu_ps(&logits[i]);
        __m256 v_min = _mm256_set1_ps(current_min);
        __m256 v_cmp = _mm256_cmp_ps(v_logits, v_min, _CMP_GT_OQ);
        int mask = _mm256_movemask_ps(v_cmp);

        // Fast branchless skip: if none of the 8 floats exceed current_min, skip immediately!
        if (__builtin_expect(mask == 0, 1)) {
            continue;
        }

        // Process elements that exceeded current_min
        for (int lane = 0; lane < 8; ++lane) {
            if (mask & (1 << lane)) {
                float val = logits[i + lane];
                if (val > current_min) {
                    top_vals[min_slot] = val;
                    top_idxs[min_slot] = i + lane;

                    // Find new minimum among top_vals
                    current_min = top_vals[0];
                    min_slot = 0;
                    for (int s = 1; s < k; ++s) {
                        if (top_vals[s] < current_min) {
                            current_min = top_vals[s];
                            min_slot = s;
                        }
                    }
                }
            }
        }
    }

    // Scalar tail for remainder elements
    for (; i < vocab_size; ++i) {
        float val = logits[i];
        if (val > current_min) {
            top_vals[min_slot] = val;
            top_idxs[min_slot] = i;
            current_min = top_vals[0];
            min_slot = 0;
            for (int s = 1; s < k; ++s) {
                if (top_vals[s] < current_min) {
                    current_min = top_vals[s];
                    min_slot = s;
                }
            }
        }
    }

    // Sort the top-k in descending order
    for (int a = 0; a < k - 1; ++a) {
        int best_idx = a;
        for (int b = a + 1; b < k; ++b) {
            if (top_vals[b] > top_vals[best_idx]) {
                best_idx = b;
            }
        }
        if (best_idx != a) {
            float tmp_v = top_vals[a];
            top_vals[a] = top_vals[best_idx];
            top_vals[best_idx] = tmp_v;

            int tmp_i = top_idxs[a];
            top_idxs[a] = top_idxs[best_idx];
            top_idxs[best_idx] = tmp_i;
        }
    }

    // Compute numerically stable Softmax over the top-k
    float max_val = top_vals[0];
    float sum_exp = 0.0f;
    for (int a = 0; a < k; ++a) {
        top_vals[a] = expf(top_vals[a] - max_val);
        sum_exp += top_vals[a];
    }

    float inv_sum = 1.0f / sum_exp;
    float entropy = 0.0f;
    for (int a = 0; a < k; ++a) {
        float p = top_vals[a] * inv_sum;
        out_probs[a] = p;
        out_indices[a] = top_idxs[a];
        if (p > 1e-12f) {
            entropy -= p * (log2f(p));
        }
    }

    if (out_entropy) *out_entropy = entropy;
    if (out_margin) {
        *out_margin = (k >= 2) ? (out_probs[0] - out_probs[1]) : 1.0f;
    }
}

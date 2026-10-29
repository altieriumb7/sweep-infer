#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

EXPORT void synaptic_tree_avx2_forward(
    const float* __restrict__ node_latents,
    const float* __restrict__ branch_weights,
    const float* __restrict__ branch_biases,
    const int*   __restrict__ parent_indices,
    int num_nodes,
    int D,
    float prob_threshold,
    float* __restrict__ out_path_scores,
    int*   __restrict__ out_selected_nodes,
    float* __restrict__ out_causal_mask,
    int*   __restrict__ out_num_selected
) {
    if (num_nodes <= 0) {
        *out_num_selected = 0;
        return;
    }

    // Allocate stack buffers for up to 32 nodes
    float logits[32];
    float scores[32];
    if (num_nodes > 32) num_nodes = 32;

    // 1. Vectorized dot products for all nodes using AVX-2 + FMA
    for (int i = 0; i < num_nodes; ++i) {
        const float* x = node_latents + (size_t)i * D;
        const float* w = branch_weights + (size_t)i * D;
        __m256 v_acc = _mm256_setzero_ps();

        int j = 0;
        for (; j <= D - 8; j += 8) {
            __m256 vx = _mm256_loadu_ps(x + j);
            __m256 vw = _mm256_loadu_ps(w + j);
            v_acc = _mm256_fmadd_ps(vx, vw, v_acc);
        }

        __m128 v_hi = _mm256_extractf128_ps(v_acc, 1);
        __m128 v_lo = _mm256_castps256_ps128(v_acc);
        __m128 v_sum = _mm_add_ps(v_lo, v_hi);
        v_sum = _mm_hadd_ps(v_sum, v_sum);
        v_sum = _mm_hadd_ps(v_sum, v_sum);
        float sum = _mm_cvtss_f32(v_sum);

        for (; j < D; ++j) {
            sum += x[j] * w[j];
        }
        logits[i] = sum + branch_biases[i];
    }

    // 2. Compute path scores hierarchically
    scores[0] = 1.0f;
    for (int i = 1; i < num_nodes; ++i) {
        int P = parent_indices[i];
        if (P < 0 || scores[P] <= 0.0f) {
            scores[i] = 0.0f;
            continue;
        }

        // Find siblings sharing parent P
        int siblings[32];
        int num_sibs = 0;
        int idx_in_sib = 0;
        for (int k = 1; k < num_nodes; ++k) {
            if (parent_indices[k] == P) {
                if (k == i) idx_in_sib = num_sibs;
                siblings[num_sibs++] = k;
            }
        }

        float p_i = 0.0f;
        if (num_sibs > 1) {
            float max_z = logits[siblings[0]];
            for (int s = 1; s < num_sibs; ++s) {
                if (logits[siblings[s]] > max_z) max_z = logits[siblings[s]];
            }
            float sum_exp = 0.0f;
            for (int s = 0; s < num_sibs; ++s) {
                sum_exp += expf(logits[siblings[s]] - max_z);
            }
            p_i = expf(logits[i] - max_z) / sum_exp;
        } else {
            p_i = 1.0f / (1.0f + expf(-logits[i]));
        }

        scores[i] = scores[P] * p_i;
    }

    // 3. Pruning against threshold
    for (int i = 0; i < num_nodes; ++i) {
        if (scores[i] < prob_threshold) {
            scores[i] = 0.0f;
        }
    }
    // Zero out descendants of pruned nodes
    for (int i = 1; i < num_nodes; ++i) {
        int P = parent_indices[i];
        if (P >= 0 && scores[P] == 0.0f) {
            scores[i] = 0.0f;
        }
    }

    // Copy to output scores
    for (int i = 0; i < num_nodes; ++i) {
        out_path_scores[i] = scores[i];
    }

    // 4. Select top surviving nodes (max 4)
    for (int i = 0; i < 4; ++i) {
        out_selected_nodes[i] = -1;
    }

    int num_sel = 0;
    if (scores[0] > 0.0f) {
        out_selected_nodes[0] = 0;
        num_sel = 1;
    }

    // Gather candidate siblings
    int candidates[32];
    int num_cands = 0;
    for (int k = 1; k < num_nodes; ++k) {
        if (scores[k] > 0.0f) {
            candidates[num_cands++] = k;
        }
    }

    // Sort candidates descending by score
    for (int a = 0; a < num_cands - 1; ++a) {
        for (int b = a + 1; b < num_cands; ++b) {
            if (scores[candidates[b]] > scores[candidates[a]]) {
                int tmp = candidates[a];
                candidates[a] = candidates[b];
                candidates[b] = tmp;
            }
        }
    }

    for (int c = 0; c < num_cands && num_sel < 4; ++c) {
        out_selected_nodes[num_sel++] = candidates[c];
    }

    *out_num_selected = num_sel;

    // 5. Construct 4x4 causal attention mask
    memset(out_causal_mask, 0, 16 * sizeof(float));
    for (int r = 0; r < num_sel; ++r) {
        int u = out_selected_nodes[r];
        for (int c = 0; c < num_sel; ++c) {
            int v = out_selected_nodes[c];
            int curr = u;
            int is_anc = 0;
            while (curr >= 0) {
                if (curr == v) {
                    is_anc = 1;
                    break;
                }
                if (curr == 0) break;
                curr = parent_indices[curr];
            }
            if (is_anc) {
                out_causal_mask[r * 4 + c] = 1.0f;
            }
        }
    }
}

// ==============================================================================
// SWEEP: Multi-SM Grid-Level CUDA Kernel for In-VRAM Top-K & Shannon Entropy
// Target: NVIDIA GeForce RTX 5070 Ti (Blackwell Architecture, SM 12.0)
//
// Stage 1: 64 threadblocks (16,384 threads across all SMs) scan vocabulary
// Stage 2: Fast Warp-level reduction across candidate pool
// ==============================================================================

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

#define WARP_SIZE 32
#define MAX_TOP_K 16
#define NUM_BLOCKS 64
#define BLOCK_SIZE 256

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Stage 1: Grid scan - Each block finds its local top-K candidates
__global__ void sweep_topk_stage1_kernel(
    const float* __restrict__ logits,
    int vocab_size,
    int k,
    float* __restrict__ block_top_vals,
    int*   __restrict__ block_top_idxs
) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    __shared__ float s_max_vals[8];
    __shared__ float s_top_vals[MAX_TOP_K];
    __shared__ int   s_top_idxs[MAX_TOP_K];

    float my_val = -1e30f;
    int my_idx = -1;

    // Grid-stride loop across vocabulary partitioned across all SMs
    int stride = gridDim.x * blockDim.x;
    for (int i = bid * blockDim.x + tid; i < vocab_size; i += stride) {
        float v = logits[i];
        if (v > my_val) {
            my_val = v;
            my_idx = i;
        }
    }

    // Find top-K inside this block
    for (int step = 0; step < k; ++step) {
        float step_max = my_val;
        int step_idx = my_idx;

        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(0xffffffff, step_max, offset);
            int other_idx = __shfl_down_sync(0xffffffff, step_idx, offset);
            if (other_val > step_max) {
                step_max = other_val;
                step_idx = other_idx;
            }
        }

        if (lane == 0) {
            s_max_vals[warp_id] = step_max;
        }
        __syncthreads();

        if (tid == 0) {
            float best_v = s_max_vals[0];
            int best_w = 0;
            for (int w = 1; w < num_warps; ++w) {
                if (s_max_vals[w] > best_v) {
                    best_v = s_max_vals[w];
                    best_w = w;
                }
            }
            s_top_vals[step] = best_v;
            s_top_idxs[step] = best_w;
        }
        __syncthreads();

        int win_warp = s_top_idxs[step];
        if (warp_id == win_warp && my_val == s_top_vals[step]) {
            s_top_idxs[step] = my_idx;
            my_val = -1e30f;
        }
        __syncthreads();
    }

    if (tid < k) {
        block_top_vals[bid * k + tid] = s_top_vals[tid];
        block_top_idxs[bid * k + tid] = s_top_idxs[tid];
    }
}

// Stage 2: Merge the NUM_BLOCKS * k candidates (e.g. 64 * 16 = 1024 items), compute Softmax & Entropy
__global__ void sweep_topk_stage2_kernel(
    const float* __restrict__ block_top_vals,
    const int*   __restrict__ block_top_idxs,
    int total_candidates,
    int k,
    float entropy_threshold,
    float margin_threshold,
    int* __restrict__ out_indices,
    float* __restrict__ out_probs,
    float* __restrict__ out_entropy,
    float* __restrict__ out_margin,
    int* __restrict__ out_is_confident
) {
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    __shared__ float s_max_vals[8];
    __shared__ float s_top_vals[MAX_TOP_K];
    __shared__ int   s_top_idxs[MAX_TOP_K];

    float my_val = -1e30f;
    int my_idx = -1;

    for (int i = tid; i < total_candidates; i += blockDim.x) {
        float v = block_top_vals[i];
        if (v > my_val) {
            my_val = v;
            my_idx = block_top_idxs[i];
        }
    }

    for (int step = 0; step < k; ++step) {
        float step_max = my_val;
        int step_idx = my_idx;

        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(0xffffffff, step_max, offset);
            int other_idx = __shfl_down_sync(0xffffffff, step_idx, offset);
            if (other_val > step_max) {
                step_max = other_val;
                step_idx = other_idx;
            }
        }

        if (lane == 0) {
            s_max_vals[warp_id] = step_max;
        }
        __syncthreads();

        if (tid == 0) {
            float best_v = s_max_vals[0];
            int best_w = 0;
            for (int w = 1; w < num_warps; ++w) {
                if (s_max_vals[w] > best_v) {
                    best_v = s_max_vals[w];
                    best_w = w;
                }
            }
            s_top_vals[step] = best_v;
            s_top_idxs[step] = best_w;
        }
        __syncthreads();

        int win_warp = s_top_idxs[step];
        if (warp_id == win_warp && my_val == s_top_vals[step]) {
            s_top_idxs[step] = step_idx;
            my_val = -1e30f;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float max_logit = s_top_vals[0];
        float sum_e = 0.0f;
        for (int i = 0; i < k; ++i) {
            float exp_val = expf(s_top_vals[i] - max_logit);
            s_top_vals[i] = exp_val;
            sum_e += exp_val;
        }

        float entropy = 0.0f;
        const float inv_ln2 = 1.4426950408889634f;
        for (int i = 0; i < k; ++i) {
            float p = s_top_vals[i] / sum_e;
            out_probs[i] = p;
            out_indices[i] = s_top_idxs[i];
            if (p > 1e-8f) {
                entropy -= p * (logf(p) * inv_ln2);
            }
        }

        float margin = (k >= 2) ? (out_probs[0] - out_probs[1]) : 1.0f;
        *out_entropy = entropy;
        *out_margin = margin;
        *out_is_confident = (entropy < entropy_threshold && margin >= margin_threshold) ? 1 : 0;
    }
}

extern "C" {

void launch_sweep_gpu_entropy_opt(
    const float* d_logits,
    int vocab_size,
    int k,
    float entropy_threshold,
    float margin_threshold,
    float* d_inter_vals,
    int*   d_inter_idxs,
    int* d_out_indices,
    float* d_out_probs,
    float* d_out_entropy,
    float* d_out_margin,
    int* d_out_is_confident,
    cudaStream_t stream
) {
    sweep_topk_stage1_kernel<<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>>(
        d_logits, vocab_size, k, d_inter_vals, d_inter_idxs
    );
    sweep_topk_stage2_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
        d_inter_vals, d_inter_idxs, NUM_BLOCKS * k, k,
        entropy_threshold, margin_threshold,
        d_out_indices, d_out_probs, d_out_entropy, d_out_margin, d_out_is_confident
    );
}

}

#pragma once

#include "common.cuh"

// H2O (Heavy Hitter Oracle) KV cache scoring kernel.
// Computes QK dot product for the most recent query against all KV entries,
// producing a per-KV importance score used for eviction decisions.
//
// This kernel runs periodically (every N decode steps) and is separate from
// the main flash-attention kernel to avoid modifying fattn's interface.

// Compute QK dot product scores for all KV entries.
// Q: [head_dim] for the latest token (single query)
// K: [n_kv, head_dim] for all KV entries (quantized or F16)
// scores: [n_kv] output - max score across all heads for each KV entry
//
// Grid: (n_kv_blocks, n_heads, 1)
// Block: (WARP_SIZE, 1, 1)
template <int Dk>
static __global__ void k_h2o_compute_scores(
        const float * __restrict__ Q,      // [n_heads, head_dim] latest query (F32)
        const half  * __restrict__ K,      // [n_kv, n_kv_heads, head_dim] (F16, contiguous)
        float       * __restrict__ scores, // [n_kv] output: max attention score per KV entry
        const int n_kv,
        const int n_heads,
        const int n_kv_heads,
        const int nb_k_row) {              // stride between KV entries in bytes

    const int kv_idx  = blockIdx.x;
    const int head    = blockIdx.y;
    const int kv_head = head / (n_heads / n_kv_heads); // GQA mapping

    if (kv_idx >= n_kv) return;

    // Q pointer for this head
    const float * Q_head = Q + head * Dk;

    // K pointer for this KV entry and KV head
    const half * K_entry = (const half *)((const char *)K + (int64_t)kv_idx * nb_k_row) + kv_head * Dk;

    // Compute dot product Q·K
    float sum = 0.0f;
    for (int i = threadIdx.x; i < Dk; i += WARP_SIZE) {
        sum += Q_head[i] * __half2float(K_entry[i]);
    }

    // Warp reduce
    sum = warp_reduce_sum(sum);

    // Thread 0 updates the max score for this KV entry (across all heads)
    if (threadIdx.x == 0) {
        atomicMax((int *)&scores[kv_idx], __float_as_int(sum));
    }
}

// Host function to compute H2O scores
void ggml_cuda_h2o_compute_scores(
        const float * Q_data,    // [n_heads * head_dim] on GPU
        const char  * K_data,    // KV cache K data on GPU
        float       * scores,    // [n_kv] output on GPU
        int n_kv,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        int nb_k_row,            // byte stride between KV entries
        cudaStream_t stream);

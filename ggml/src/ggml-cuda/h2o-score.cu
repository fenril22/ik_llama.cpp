#include "h2o-score.cuh"

// atomicMax for float (using int reinterpretation)
// Note: only works correctly for positive floats or when used with __float_as_int
// which preserves ordering for positive values.

void ggml_cuda_h2o_compute_scores(
        const float * Q_data,
        const char  * K_data,
        float       * scores,
        int n_kv,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        int nb_k_row,
        cudaStream_t stream) {

    // Clear scores to -infinity
    const int n_bytes = n_kv * sizeof(float);
    // Use a large negative value as int representation
    CUDA_CHECK(cudaMemsetAsync(scores, 0x00, n_bytes, stream)); // 0.0f as baseline

    const dim3 blocks(n_kv, n_heads, 1);
    const dim3 threads(WARP_SIZE, 1, 1);

    switch (head_dim) {
        case 64:
            k_h2o_compute_scores<64><<<blocks, threads, 0, stream>>>(
                Q_data, (const half *)K_data, scores, n_kv, n_heads, n_kv_heads, nb_k_row);
            break;
        case 128:
            k_h2o_compute_scores<128><<<blocks, threads, 0, stream>>>(
                Q_data, (const half *)K_data, scores, n_kv, n_heads, n_kv_heads, nb_k_row);
            break;
        case 256:
            k_h2o_compute_scores<256><<<blocks, threads, 0, stream>>>(
                Q_data, (const half *)K_data, scores, n_kv, n_heads, n_kv_heads, nb_k_row);
            break;
        default:
            GGML_ASSERT(false && "Unsupported head_dim for H2O scoring");
    }

    CUDA_CHECK(cudaGetLastError());
}

/*
 * CUDA kernels for F16/F32 → TurboQuant bulk conversion.
 *
 * Input is already WHT-rotated (by ggml_turbo_wht op in the graph).
 * Normalization method: L2 norm (same as reference/CPU path).
 * WHT post-rotation values follow N(0, 1/d), so L2 norm aligns with
 * the Lloyd-Max centroid distribution assumption.
 *
 * Kernel structure: 1 block per QK group, QK threads per block.
 * Warp reduction for L2 norm and reconstruction norm.
 * Warp-cooperative bit packing (no atomics).
 */

#include "common.cuh"
#include "ggml-common.h"
#include "turbo-quant.cuh"

// ── Helpers ───────────────────────────────────────────────────────────

static __device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, offset);
    }
    return v;
}

template <int N_WARPS>
static __device__ __forceinline__ float block_reduce_sum(float v, float * smem_accum) {
    const int lane   = threadIdx.x % WARP_SIZE;
    const int warp   = threadIdx.x / WARP_SIZE;
    v = warp_reduce_sum(v);
    if (lane == 0) smem_accum[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = (lane < N_WARPS) ? smem_accum[lane] : 0.0f;
        v = warp_reduce_sum(v);
    }
    __syncthreads();
    return v;
}

// ── Encode: F16/F32 → turbo3 ─────────────────────────────────────────
//
// Layout: 1 block = 1 group of QK_TURBO3 elements, QK_TURBO3 threads.
// n_groups = total_elements / QK_TURBO3.

static __global__ void k_cpy_f16_turbo3(
        const half * __restrict__ s,
        block_turbo3_0 * __restrict__ d,
        int n_blocks) {

    constexpr int N = QK_TURBO3;
    constexpr int N_WARPS = N / WARP_SIZE;

    const int b = blockIdx.x;
    if (b >= n_blocks) return;
    const int j = threadIdx.x;  // 0..N-1

    __shared__ float x[N];
    __shared__ float smem[N_WARPS];

    x[j] = __half2float(s[b * N + j]);
    __syncthreads();

    // L2 norm (warp reduction)
    float norm_sq = block_reduce_sum<N_WARPS>(x[j] * x[j], smem);
    const float grp_norm = sqrtf(norm_sq);
    const float inv = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    const float xn = x[j] * inv;
    const uint8_t idx = turbo_nearest_centroid_3bit(xn);

    // Pack qs (4 elements per byte, 2-bit each) — warp cooperative
    const int qs_byte = j / 4;
    const uint8_t my_low2 = idx & 0x3;
    uint8_t qs_byte_val = 0;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        qs_byte_val |= __shfl_sync(0xffffffff, my_low2, (j & ~3) + k) << (k * 2);
    }
    if (j % 4 == 0) d[b].qs[qs_byte] = qs_byte_val;

    // Pack signs (8 elements per byte) — ballot
    const uint32_t ballot = __ballot_sync(0xffffffff, (idx >> 2) & 1);
    const int signs_byte = j / 8;
    if (j % 8 == 0) d[b].signs[signs_byte] = (uint8_t)((ballot >> ((j % WARP_SIZE) / 8 * 8)) & 0xFF);

    // Reconstruction norm (warp reduction)
    const float c = TURBO_CENTROIDS_3BIT[idx];
    float recon_sq = block_reduce_sum<N_WARPS>(c * c, smem);
    if (j == 0) {
        const float rn = sqrtf(recon_sq);
        d[b].norm = __float2half((rn > 1e-10f) ? grp_norm / rn : grp_norm);
    }
}

static __global__ void k_cpy_f32_turbo3(
        const float * __restrict__ s,
        block_turbo3_0 * __restrict__ d,
        int n_blocks) {

    constexpr int N = QK_TURBO3;
    constexpr int N_WARPS = N / WARP_SIZE;

    const int b = blockIdx.x;
    if (b >= n_blocks) return;
    const int j = threadIdx.x;

    __shared__ float smem[N_WARPS];

    const float v = s[b * N + j];

    float norm_sq = block_reduce_sum<N_WARPS>(v * v, smem);
    const float grp_norm = sqrtf(norm_sq);
    const float inv = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    const float xn = v * inv;
    const uint8_t idx = turbo_nearest_centroid_3bit(xn);

    const int qs_byte = j / 4;
    const uint8_t my_low2 = idx & 0x3;
    uint8_t qs_byte_val = 0;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        qs_byte_val |= __shfl_sync(0xffffffff, my_low2, (j & ~3) + k) << (k * 2);
    }
    if (j % 4 == 0) d[b].qs[qs_byte] = qs_byte_val;

    const uint32_t ballot = __ballot_sync(0xffffffff, (idx >> 2) & 1);
    const int signs_byte = j / 8;
    if (j % 8 == 0) d[b].signs[signs_byte] = (uint8_t)((ballot >> ((j % WARP_SIZE) / 8 * 8)) & 0xFF);

    const float c = TURBO_CENTROIDS_3BIT[idx];
    float recon_sq = block_reduce_sum<N_WARPS>(c * c, smem);
    if (j == 0) {
        const float rn = sqrtf(recon_sq);
        d[b].norm = __float2half((rn > 1e-10f) ? grp_norm / rn : grp_norm);
    }
}

// ── Encode: F16/F32 → turbo4 ─────────────────────────────────────────

static __global__ void k_cpy_f16_turbo4(
        const half * __restrict__ s,
        block_turbo4_0 * __restrict__ d,
        int n_blocks) {

    constexpr int N = QK_TURBO4;
    constexpr int N_WARPS = N / WARP_SIZE;

    const int b = blockIdx.x;
    if (b >= n_blocks) return;
    const int j = threadIdx.x;

    __shared__ float smem[N_WARPS];

    const float v = __half2float(s[b * N + j]);

    float norm_sq = block_reduce_sum<N_WARPS>(v * v, smem);
    const float grp_norm = sqrtf(norm_sq);
    const float inv = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    const uint8_t idx = turbo_nearest_centroid_4bit(v * inv);

    // Pack qs: nibble-packed (2 elements per byte)
    const uint8_t my_nibble = idx & 0xF;
    uint8_t qs_byte_val = __shfl_sync(0xffffffff, my_nibble, j & ~1) |
                          (__shfl_sync(0xffffffff, my_nibble, j | 1) << 4);
    if (j % 2 == 0) d[b].qs[j / 2] = qs_byte_val;

    const float c = TURBO_CENTROIDS_4BIT[idx];
    float recon_sq = block_reduce_sum<N_WARPS>(c * c, smem);
    if (j == 0) {
        const float rn = sqrtf(recon_sq);
        d[b].norm  = __float2half((rn > 1e-10f) ? grp_norm / rn : grp_norm);
        d[b].rnorm = __float2half(0.0f);
    }
}

static __global__ void k_cpy_f32_turbo4(
        const float * __restrict__ s,
        block_turbo4_0 * __restrict__ d,
        int n_blocks) {

    constexpr int N = QK_TURBO4;
    constexpr int N_WARPS = N / WARP_SIZE;

    const int b = blockIdx.x;
    if (b >= n_blocks) return;
    const int j = threadIdx.x;

    __shared__ float smem[N_WARPS];

    const float v = s[b * N + j];

    float norm_sq = block_reduce_sum<N_WARPS>(v * v, smem);
    const float grp_norm = sqrtf(norm_sq);
    const float inv = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    const uint8_t idx = turbo_nearest_centroid_4bit(v * inv);

    const uint8_t my_nibble = idx & 0xF;
    uint8_t qs_byte_val = __shfl_sync(0xffffffff, my_nibble, j & ~1) |
                          (__shfl_sync(0xffffffff, my_nibble, j | 1) << 4);
    if (j % 2 == 0) d[b].qs[j / 2] = qs_byte_val;

    const float c = TURBO_CENTROIDS_4BIT[idx];
    float recon_sq = block_reduce_sum<N_WARPS>(c * c, smem);
    if (j == 0) {
        const float rn = sqrtf(recon_sq);
        d[b].norm  = __float2half((rn > 1e-10f) ? grp_norm / rn : grp_norm);
        d[b].rnorm = __float2half(0.0f);
    }
}

// ── Encode: F16/F32 → turbo2 ─────────────────────────────────────────

static __global__ void k_cpy_f16_turbo2(
        const half * __restrict__ s,
        block_turbo2_0 * __restrict__ d,
        int n_blocks) {

    constexpr int N = QK_TURBO2;
    constexpr int N_WARPS = N / WARP_SIZE;

    const int b = blockIdx.x;
    if (b >= n_blocks) return;
    const int j = threadIdx.x;

    __shared__ float smem[N_WARPS];

    const float v = __half2float(s[b * N + j]);

    float norm_sq = block_reduce_sum<N_WARPS>(v * v, smem);
    const float grp_norm = sqrtf(norm_sq);
    const float inv = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    const uint8_t idx = turbo_nearest_centroid_2bit(v * inv);

    // Pack qs: 4 elements per byte, 2-bit each
    const uint8_t my_low2 = idx & 0x3;
    uint8_t qs_byte_val = 0;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        qs_byte_val |= __shfl_sync(0xffffffff, my_low2, (j & ~3) + k) << (k * 2);
    }
    if (j % 4 == 0) d[b].qs[j / 4] = qs_byte_val;

    const float c = TURBO_CENTROIDS_2BIT[idx];
    float recon_sq = block_reduce_sum<N_WARPS>(c * c, smem);
    if (j == 0) {
        const float rn = sqrtf(recon_sq);
        d[b].norm = __float2half((rn > 1e-10f) ? grp_norm / rn : grp_norm);
    }
}

static __global__ void k_cpy_f32_turbo2(
        const float * __restrict__ s,
        block_turbo2_0 * __restrict__ d,
        int n_blocks) {

    constexpr int N = QK_TURBO2;
    constexpr int N_WARPS = N / WARP_SIZE;

    const int b = blockIdx.x;
    if (b >= n_blocks) return;
    const int j = threadIdx.x;

    __shared__ float smem[N_WARPS];

    const float v = s[b * N + j];

    float norm_sq = block_reduce_sum<N_WARPS>(v * v, smem);
    const float grp_norm = sqrtf(norm_sq);
    const float inv = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    const uint8_t idx = turbo_nearest_centroid_2bit(v * inv);

    const uint8_t my_low2 = idx & 0x3;
    uint8_t qs_byte_val = 0;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        qs_byte_val |= __shfl_sync(0xffffffff, my_low2, (j & ~3) + k) << (k * 2);
    }
    if (j % 4 == 0) d[b].qs[j / 4] = qs_byte_val;

    const float c = TURBO_CENTROIDS_2BIT[idx];
    float recon_sq = block_reduce_sum<N_WARPS>(c * c, smem);
    if (j == 0) {
        const float rn = sqrtf(recon_sq);
        d[b].norm = __float2half((rn > 1e-10f) ? grp_norm / rn : grp_norm);
    }
}

// ── Public dispatch functions ─────────────────────────────────────────

void ggml_cuda_cpy_f16_turbo3(const char * cx, char * cdst, const int ne, cudaStream_t stream) {
    const int n_blocks = ne / QK_TURBO3;
    k_cpy_f16_turbo3<<<n_blocks, QK_TURBO3, 0, stream>>>((const half *)cx, (block_turbo3_0 *)cdst, n_blocks);
}

void ggml_cuda_cpy_f32_turbo3(const char * cx, char * cdst, const int ne, cudaStream_t stream) {
    const int n_blocks = ne / QK_TURBO3;
    k_cpy_f32_turbo3<<<n_blocks, QK_TURBO3, 0, stream>>>((const float *)cx, (block_turbo3_0 *)cdst, n_blocks);
}

void ggml_cuda_cpy_f16_turbo4(const char * cx, char * cdst, const int ne, cudaStream_t stream) {
    const int n_blocks = ne / QK_TURBO4;
    k_cpy_f16_turbo4<<<n_blocks, QK_TURBO4, 0, stream>>>((const half *)cx, (block_turbo4_0 *)cdst, n_blocks);
}

void ggml_cuda_cpy_f32_turbo4(const char * cx, char * cdst, const int ne, cudaStream_t stream) {
    const int n_blocks = ne / QK_TURBO4;
    k_cpy_f32_turbo4<<<n_blocks, QK_TURBO4, 0, stream>>>((const float *)cx, (block_turbo4_0 *)cdst, n_blocks);
}

void ggml_cuda_cpy_f16_turbo2(const char * cx, char * cdst, const int ne, cudaStream_t stream) {
    const int n_blocks = ne / QK_TURBO2;
    k_cpy_f16_turbo2<<<n_blocks, QK_TURBO2, 0, stream>>>((const half *)cx, (block_turbo2_0 *)cdst, n_blocks);
}

void ggml_cuda_cpy_f32_turbo2(const char * cx, char * cdst, const int ne, cudaStream_t stream) {
    const int n_blocks = ne / QK_TURBO2;
    k_cpy_f32_turbo2<<<n_blocks, QK_TURBO2, 0, stream>>>((const float *)cx, (block_turbo2_0 *)cdst, n_blocks);
}

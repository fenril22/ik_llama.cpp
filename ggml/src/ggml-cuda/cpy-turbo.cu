/*
 * CUDA kernels for F16/F32 → TurboQuant bulk conversion.
 * Input is already WHT-rotated (by ggml_turbo_wht op in the graph).
 * Only absmax normalization + centroid quantization needed here.
 */

#include "common.cuh"
#include "ggml-common.h"
#include "turbo-quant.cuh"

// ── Encode: F16 → turbo3 ─────────────────────────────────────────────

static __global__ void k_cpy_f16_turbo3(const half * __restrict__ s, block_turbo3_0 * __restrict__ d, int n_blocks) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;

    float buf[QK_TURBO3];
    float grp_norm = 0.0f;
    #pragma unroll
    for (int j = 0; j < QK_TURBO3; j++) {
        buf[j] = __half2float(s[b * QK_TURBO3 + j]);
        float av = fabsf(buf[j]);
        if (av > grp_norm) grp_norm = av;
    }
    float inv = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

    for (int j = 0; j < QK_TURBO3 / 4; j++) d[b].qs[j] = 0;
    for (int j = 0; j < QK_TURBO3 / 8; j++) d[b].signs[j] = 0;

    float recon_sq = 0.0f;
    for (int j = 0; j < QK_TURBO3; j++) {
        uint8_t idx = turbo_nearest_centroid_3bit(buf[j] * inv);
        d[b].qs[j / 4]    |= (idx & 0x3) << ((j % 4) * 2);
        if (idx & 0x4) d[b].signs[j / 8] |= (1 << (j % 8));
        float c = TURBO_CENTROIDS_3BIT[idx];
        recon_sq += c * c;
    }
    float rn = sqrtf(recon_sq);
    d[b].norm = __float2half((rn > 1e-10f) ? grp_norm / rn : grp_norm);
}

// ── Encode: F32 → turbo3 ─────────────────────────────────────────────

static __global__ void k_cpy_f32_turbo3(const float * __restrict__ s, block_turbo3_0 * __restrict__ d, int n_blocks) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;

    float grp_norm = 0.0f;
    #pragma unroll
    for (int j = 0; j < QK_TURBO3; j++) {
        float av = fabsf(s[b * QK_TURBO3 + j]);
        if (av > grp_norm) grp_norm = av;
    }
    float inv = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

    for (int j = 0; j < QK_TURBO3 / 4; j++) d[b].qs[j] = 0;
    for (int j = 0; j < QK_TURBO3 / 8; j++) d[b].signs[j] = 0;

    float recon_sq = 0.0f;
    for (int j = 0; j < QK_TURBO3; j++) {
        uint8_t idx = turbo_nearest_centroid_3bit(s[b * QK_TURBO3 + j] * inv);
        d[b].qs[j / 4]    |= (idx & 0x3) << ((j % 4) * 2);
        if (idx & 0x4) d[b].signs[j / 8] |= (1 << (j % 8));
        float c = TURBO_CENTROIDS_3BIT[idx];
        recon_sq += c * c;
    }
    float rn = sqrtf(recon_sq);
    d[b].norm = __float2half((rn > 1e-10f) ? grp_norm / rn : grp_norm);
}

// ── Encode: F16 → turbo4 ─────────────────────────────────────────────

static __global__ void k_cpy_f16_turbo4(const half * __restrict__ s, block_turbo4_0 * __restrict__ d, int n_blocks) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;

    float buf[QK_TURBO4];
    float grp_norm = 0.0f;
    #pragma unroll
    for (int j = 0; j < QK_TURBO4; j++) {
        buf[j] = __half2float(s[b * QK_TURBO4 + j]);
        float av = fabsf(buf[j]);
        if (av > grp_norm) grp_norm = av;
    }
    float inv = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

    for (int j = 0; j < QK_TURBO4 / 2; j++) d[b].qs[j] = 0;

    float recon_sq = 0.0f;
    for (int j = 0; j < QK_TURBO4; j++) {
        uint8_t idx = turbo_nearest_centroid_4bit(buf[j] * inv);
        d[b].qs[j / 2] |= (idx & 0xF) << ((j % 2) * 4);
        float c = TURBO_CENTROIDS_4BIT[idx];
        recon_sq += c * c;
    }
    float rn = sqrtf(recon_sq);
    d[b].norm  = __float2half((rn > 1e-10f) ? grp_norm / rn : grp_norm);
    d[b].rnorm = __float2half(0.0f);
}

// ── Encode: F32 → turbo4 ─────────────────────────────────────────────

static __global__ void k_cpy_f32_turbo4(const float * __restrict__ s, block_turbo4_0 * __restrict__ d, int n_blocks) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;

    float grp_norm = 0.0f;
    #pragma unroll
    for (int j = 0; j < QK_TURBO4; j++) {
        float av = fabsf(s[b * QK_TURBO4 + j]);
        if (av > grp_norm) grp_norm = av;
    }
    float inv = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

    for (int j = 0; j < QK_TURBO4 / 2; j++) d[b].qs[j] = 0;

    float recon_sq = 0.0f;
    for (int j = 0; j < QK_TURBO4; j++) {
        uint8_t idx = turbo_nearest_centroid_4bit(s[b * QK_TURBO4 + j] * inv);
        d[b].qs[j / 2] |= (idx & 0xF) << ((j % 2) * 4);
        float c = TURBO_CENTROIDS_4BIT[idx];
        recon_sq += c * c;
    }
    float rn = sqrtf(recon_sq);
    d[b].norm  = __float2half((rn > 1e-10f) ? grp_norm / rn : grp_norm);
    d[b].rnorm = __float2half(0.0f);
}

// ── Encode: F16 → turbo2 ─────────────────────────────────────────────

static __global__ void k_cpy_f16_turbo2(const half * __restrict__ s, block_turbo2_0 * __restrict__ d, int n_blocks) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;

    float buf[QK_TURBO2];
    float grp_norm = 0.0f;
    #pragma unroll
    for (int j = 0; j < QK_TURBO2; j++) {
        buf[j] = __half2float(s[b * QK_TURBO2 + j]);
        float av = fabsf(buf[j]);
        if (av > grp_norm) grp_norm = av;
    }
    float inv = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

    for (int j = 0; j < QK_TURBO2 / 4; j++) d[b].qs[j] = 0;

    float recon_sq = 0.0f;
    for (int j = 0; j < QK_TURBO2; j++) {
        uint8_t idx = turbo_nearest_centroid_2bit(buf[j] * inv);
        d[b].qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
        float c = TURBO_CENTROIDS_2BIT[idx];
        recon_sq += c * c;
    }
    float rn = sqrtf(recon_sq);
    d[b].norm = __float2half((rn > 1e-10f) ? grp_norm / rn : grp_norm);
}

// ── Encode: F32 → turbo2 ─────────────────────────────────────────────

static __global__ void k_cpy_f32_turbo2(const float * __restrict__ s, block_turbo2_0 * __restrict__ d, int n_blocks) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;

    float grp_norm = 0.0f;
    #pragma unroll
    for (int j = 0; j < QK_TURBO2; j++) {
        float av = fabsf(s[b * QK_TURBO2 + j]);
        if (av > grp_norm) grp_norm = av;
    }
    float inv = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

    for (int j = 0; j < QK_TURBO2 / 4; j++) d[b].qs[j] = 0;

    float recon_sq = 0.0f;
    for (int j = 0; j < QK_TURBO2; j++) {
        uint8_t idx = turbo_nearest_centroid_2bit(s[b * QK_TURBO2 + j] * inv);
        d[b].qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
        float c = TURBO_CENTROIDS_2BIT[idx];
        recon_sq += c * c;
    }
    float rn = sqrtf(recon_sq);
    d[b].norm = __float2half((rn > 1e-10f) ? grp_norm / rn : grp_norm);
}

// ── Public dispatch functions ─────────────────────────────────────────

static constexpr int CPY_TURBO_THREADS = 256;

void ggml_cuda_cpy_f16_turbo3(const char * cx, char * cdst, const int ne, cudaStream_t stream) {
    const int n_blocks = ne / QK_TURBO3;
    const int grid = (n_blocks + CPY_TURBO_THREADS - 1) / CPY_TURBO_THREADS;
    k_cpy_f16_turbo3<<<grid, CPY_TURBO_THREADS, 0, stream>>>((const half *)cx, (block_turbo3_0 *)cdst, n_blocks);
}

void ggml_cuda_cpy_f32_turbo3(const char * cx, char * cdst, const int ne, cudaStream_t stream) {
    const int n_blocks = ne / QK_TURBO3;
    const int grid = (n_blocks + CPY_TURBO_THREADS - 1) / CPY_TURBO_THREADS;
    k_cpy_f32_turbo3<<<grid, CPY_TURBO_THREADS, 0, stream>>>((const float *)cx, (block_turbo3_0 *)cdst, n_blocks);
}

void ggml_cuda_cpy_f16_turbo4(const char * cx, char * cdst, const int ne, cudaStream_t stream) {
    const int n_blocks = ne / QK_TURBO4;
    const int grid = (n_blocks + CPY_TURBO_THREADS - 1) / CPY_TURBO_THREADS;
    k_cpy_f16_turbo4<<<grid, CPY_TURBO_THREADS, 0, stream>>>((const half *)cx, (block_turbo4_0 *)cdst, n_blocks);
}

void ggml_cuda_cpy_f32_turbo4(const char * cx, char * cdst, const int ne, cudaStream_t stream) {
    const int n_blocks = ne / QK_TURBO4;
    const int grid = (n_blocks + CPY_TURBO_THREADS - 1) / CPY_TURBO_THREADS;
    k_cpy_f32_turbo4<<<grid, CPY_TURBO_THREADS, 0, stream>>>((const float *)cx, (block_turbo4_0 *)cdst, n_blocks);
}

void ggml_cuda_cpy_f16_turbo2(const char * cx, char * cdst, const int ne, cudaStream_t stream) {
    const int n_blocks = ne / QK_TURBO2;
    const int grid = (n_blocks + CPY_TURBO_THREADS - 1) / CPY_TURBO_THREADS;
    k_cpy_f16_turbo2<<<grid, CPY_TURBO_THREADS, 0, stream>>>((const half *)cx, (block_turbo2_0 *)cdst, n_blocks);
}

void ggml_cuda_cpy_f32_turbo2(const char * cx, char * cdst, const int ne, cudaStream_t stream) {
    const int n_blocks = ne / QK_TURBO2;
    const int grid = (n_blocks + CPY_TURBO_THREADS - 1) / CPY_TURBO_THREADS;
    k_cpy_f32_turbo2<<<grid, CPY_TURBO_THREADS, 0, stream>>>((const float *)cx, (block_turbo2_0 *)cdst, n_blocks);
}

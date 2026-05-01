/*
 * CUDA kernels for F16/F32 → PlanarQuant/IsoQuant bulk conversion,
 * and PlanarQuant/IsoQuant → F16 dequantize for the fattn-mma path.
 *
 * All rotation constants are shared from planar-iso-constants.cuh
 * (compile-time baked __constant__ arrays, no runtime init needed).
 *
 * Four encode conversions: F16/F32 → planar3/4, iso3/4
 * Four decode conversions: planar3/4, iso3/4 → F16
 *
 * Encode: read F16/F32 → normalize → rotate → quantize → pack
 * Decode: unpack → dequantize → inverse-rotate → scale by norm
 */

#include "common.cuh"
#include "ggml-common.h"
#include "planar-iso-constants.cuh"

// ── Device helpers ───────────────────────────────────────────────────

__device__ __forceinline__ uint8_t quantize_3bit(float val) {
    if      (val < PI_MID_3BIT[0]) return 0;
    else if (val < PI_MID_3BIT[1]) return 1;
    else if (val < PI_MID_3BIT[2]) return 2;
    else if (val < PI_MID_3BIT[3]) return 3;
    else if (val < PI_MID_3BIT[4]) return 4;
    else if (val < PI_MID_3BIT[5]) return 5;
    else if (val < PI_MID_3BIT[6]) return 6;
    else                           return 7;
}

__device__ __forceinline__ uint8_t quantize_4bit(float val) {
    if      (val < PI_MID_4BIT[0])  return 0;
    else if (val < PI_MID_4BIT[1])  return 1;
    else if (val < PI_MID_4BIT[2])  return 2;
    else if (val < PI_MID_4BIT[3])  return 3;
    else if (val < PI_MID_4BIT[4])  return 4;
    else if (val < PI_MID_4BIT[5])  return 5;
    else if (val < PI_MID_4BIT[6])  return 6;
    else if (val < PI_MID_4BIT[7])  return 7;
    else if (val < PI_MID_4BIT[8])  return 8;
    else if (val < PI_MID_4BIT[9])  return 9;
    else if (val < PI_MID_4BIT[10]) return 10;
    else if (val < PI_MID_4BIT[11]) return 11;
    else if (val < PI_MID_4BIT[12]) return 12;
    else if (val < PI_MID_4BIT[13]) return 13;
    else if (val < PI_MID_4BIT[14]) return 14;
    else                            return 15;
}

// ── Encode: Planar3 F16 → block_planar3_0 ───────────────────────────

__global__ void kernel_cpy_f16_planar3(
    const half * __restrict__ src,
    block_planar3_0 * __restrict__ dst,
    int64_t n_blocks)
{
    const int64_t ib = blockIdx.x * blockDim.x + threadIdx.x;
    if (ib >= n_blocks) return;

    const half * s = src + ib * QK_PLANAR3;
    block_planar3_0 * blk = &dst[ib];

    float buf[128];
    float norm_sq = 0.0f;
    #pragma unroll
    for (int j = 0; j < QK_PLANAR3; j++) {
        buf[j] = __half2float(s[j]);
        norm_sq += buf[j] * buf[j];
    }
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    #pragma unroll
    for (int j = 0; j < QK_PLANAR3; j++) buf[j] *= inv_norm;

    float rotated[128];
    #pragma unroll
    for (int p = 0; p < 64; p++) {
        float c = PI_COS[p], sv = PI_SIN[p];
        rotated[p*2]   = c * buf[p*2] - sv * buf[p*2+1];
        rotated[p*2+1] = sv * buf[p*2] + c * buf[p*2+1];
    }

    #pragma unroll
    for (int j = 0; j < QK_PLANAR3/4; j++) blk->qs[j] = 0;
    #pragma unroll
    for (int j = 0; j < QK_PLANAR3/8; j++) blk->signs[j] = 0;

    float recon_sq = 0.0f;
    #pragma unroll
    for (int j = 0; j < QK_PLANAR3; j++) {
        uint8_t idx = quantize_3bit(rotated[j]);
        blk->qs[j/4] |= (idx & 0x3) << ((j%4)*2);
        if (idx & 0x4) blk->signs[j/8] |= (1 << (j%8));
        recon_sq += PI_CENTROIDS_3BIT[idx] * PI_CENTROIDS_3BIT[idx];
    }

    float recon_norm = sqrtf(recon_sq);
    blk->norm = __float2half(recon_norm > 1e-10f ? grp_norm / recon_norm : grp_norm);
}

// ── Encode: Planar4 F16 → block_planar4_0 ───────────────────────────

__global__ void kernel_cpy_f16_planar4(
    const half * __restrict__ src,
    block_planar4_0 * __restrict__ dst,
    int64_t n_blocks)
{
    const int64_t ib = blockIdx.x * blockDim.x + threadIdx.x;
    if (ib >= n_blocks) return;

    const half * s = src + ib * QK_PLANAR4;
    block_planar4_0 * blk = &dst[ib];

    float buf[128];
    float norm_sq = 0.0f;
    #pragma unroll
    for (int j = 0; j < QK_PLANAR4; j++) {
        buf[j] = __half2float(s[j]);
        norm_sq += buf[j] * buf[j];
    }
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    #pragma unroll
    for (int j = 0; j < QK_PLANAR4; j++) buf[j] *= inv_norm;

    float rotated[128];
    #pragma unroll
    for (int p = 0; p < 64; p++) {
        float c = PI_COS[p], sv = PI_SIN[p];
        rotated[p*2]   = c * buf[p*2] - sv * buf[p*2+1];
        rotated[p*2+1] = sv * buf[p*2] + c * buf[p*2+1];
    }

    #pragma unroll
    for (int j = 0; j < 64; j++) blk->qs[j] = 0;
    float recon_sq = 0.0f;
    #pragma unroll
    for (int j = 0; j < 128; j++) {
        uint8_t idx = quantize_4bit(rotated[j]);
        blk->qs[j/2] |= (idx & 0xF) << ((j%2)*4);
        recon_sq += PI_CENTROIDS_4BIT[idx] * PI_CENTROIDS_4BIT[idx];
    }

    float recon_norm = sqrtf(recon_sq);
    blk->norm  = __float2half(recon_norm > 1e-10f ? grp_norm / recon_norm : grp_norm);
    blk->rnorm = __float2half(0.0f);
}

// ── Encode: Iso3 F16 → block_iso3_0 ─────────────────────────────────

__global__ void kernel_cpy_f16_iso3(
    const half * __restrict__ src,
    block_iso3_0 * __restrict__ dst,
    int64_t n_blocks)
{
    const int64_t ib = blockIdx.x * blockDim.x + threadIdx.x;
    if (ib >= n_blocks) return;

    const half * s = src + ib * QK_ISO3;
    block_iso3_0 * blk = &dst[ib];

    float buf[128];
    float norm_sq = 0.0f;
    #pragma unroll
    for (int j = 0; j < QK_ISO3; j++) {
        buf[j] = __half2float(s[j]);
        norm_sq += buf[j] * buf[j];
    }
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    #pragma unroll
    for (int j = 0; j < QK_ISO3; j++) buf[j] *= inv_norm;

    /* Forward quaternion rotation per 4D group: rotated = q_L * v */
    float rotated[128];
    #pragma unroll
    for (int g = 0; g < 32; g++) {
        float qw = PI_QW[g], qx = PI_QX[g], qy = PI_QY[g], qz = PI_QZ[g];
        float v0 = buf[g*4], v1 = buf[g*4+1], v2 = buf[g*4+2], v3 = buf[g*4+3];
        rotated[g*4]   = qw*v0 - qx*v1 - qy*v2 - qz*v3;
        rotated[g*4+1] = qw*v1 + qx*v0 + qy*v3 - qz*v2;
        rotated[g*4+2] = qw*v2 - qx*v3 + qy*v0 + qz*v1;
        rotated[g*4+3] = qw*v3 + qx*v2 - qy*v1 + qz*v0;
    }

    #pragma unroll
    for (int j = 0; j < QK_ISO3/4; j++) blk->qs[j] = 0;
    #pragma unroll
    for (int j = 0; j < QK_ISO3/8; j++) blk->signs[j] = 0;

    float recon_sq = 0.0f;
    #pragma unroll
    for (int j = 0; j < QK_ISO3; j++) {
        uint8_t idx = quantize_3bit(rotated[j]);
        blk->qs[j/4] |= (idx & 0x3) << ((j%4)*2);
        if (idx & 0x4) blk->signs[j/8] |= (1 << (j%8));
        recon_sq += PI_CENTROIDS_3BIT[idx] * PI_CENTROIDS_3BIT[idx];
    }

    float recon_norm = sqrtf(recon_sq);
    blk->norm = __float2half(recon_norm > 1e-10f ? grp_norm / recon_norm : grp_norm);
}

// ── Encode: Iso4 F16 → block_iso4_0 ─────────────────────────────────

__global__ void kernel_cpy_f16_iso4(
    const half * __restrict__ src,
    block_iso4_0 * __restrict__ dst,
    int64_t n_blocks)
{
    const int64_t ib = blockIdx.x * blockDim.x + threadIdx.x;
    if (ib >= n_blocks) return;

    const half * s = src + ib * QK_ISO4;
    block_iso4_0 * blk = &dst[ib];

    float buf[128];
    float norm_sq = 0.0f;
    #pragma unroll
    for (int j = 0; j < QK_ISO4; j++) {
        buf[j] = __half2float(s[j]);
        norm_sq += buf[j] * buf[j];
    }
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    #pragma unroll
    for (int j = 0; j < QK_ISO4; j++) buf[j] *= inv_norm;

    float rotated[128];
    #pragma unroll
    for (int g = 0; g < 32; g++) {
        float qw = PI_QW[g], qx = PI_QX[g], qy = PI_QY[g], qz = PI_QZ[g];
        float v0 = buf[g*4], v1 = buf[g*4+1], v2 = buf[g*4+2], v3 = buf[g*4+3];
        rotated[g*4]   = qw*v0 - qx*v1 - qy*v2 - qz*v3;
        rotated[g*4+1] = qw*v1 + qx*v0 + qy*v3 - qz*v2;
        rotated[g*4+2] = qw*v2 - qx*v3 + qy*v0 + qz*v1;
        rotated[g*4+3] = qw*v3 + qx*v2 - qy*v1 + qz*v0;
    }

    #pragma unroll
    for (int j = 0; j < 64; j++) blk->qs[j] = 0;
    float recon_sq = 0.0f;
    #pragma unroll
    for (int j = 0; j < 128; j++) {
        uint8_t idx = quantize_4bit(rotated[j]);
        blk->qs[j/2] |= (idx & 0xF) << ((j%2)*4);
        recon_sq += PI_CENTROIDS_4BIT[idx] * PI_CENTROIDS_4BIT[idx];
    }

    float recon_norm = sqrtf(recon_sq);
    blk->norm  = __float2half(recon_norm > 1e-10f ? grp_norm / recon_norm : grp_norm);
    blk->rnorm = __float2half(0.0f);
}

// ── Encode: Planar3 F32 → block_planar3_0 ───────────────────────────

__global__ void kernel_cpy_f32_planar3(
    const float * __restrict__ src,
    block_planar3_0 * __restrict__ dst,
    int64_t n_blocks)
{
    const int64_t ib = blockIdx.x * blockDim.x + threadIdx.x;
    if (ib >= n_blocks) return;

    const float * s = src + ib * QK_PLANAR3;
    block_planar3_0 * blk = &dst[ib];

    float buf[128];
    float norm_sq = 0.0f;
    #pragma unroll
    for (int j = 0; j < QK_PLANAR3; j++) {
        buf[j] = s[j];
        norm_sq += buf[j] * buf[j];
    }
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    #pragma unroll
    for (int j = 0; j < QK_PLANAR3; j++) buf[j] *= inv_norm;

    float rotated[128];
    #pragma unroll
    for (int p = 0; p < 64; p++) {
        float c = PI_COS[p], sv = PI_SIN[p];
        rotated[p*2]   = c * buf[p*2] - sv * buf[p*2+1];
        rotated[p*2+1] = sv * buf[p*2] + c * buf[p*2+1];
    }

    #pragma unroll
    for (int j = 0; j < QK_PLANAR3/4; j++) blk->qs[j] = 0;
    #pragma unroll
    for (int j = 0; j < QK_PLANAR3/8; j++) blk->signs[j] = 0;

    float recon_sq = 0.0f;
    #pragma unroll
    for (int j = 0; j < QK_PLANAR3; j++) {
        uint8_t idx = quantize_3bit(rotated[j]);
        blk->qs[j/4] |= (idx & 0x3) << ((j%4)*2);
        if (idx & 0x4) blk->signs[j/8] |= (1 << (j%8));
        recon_sq += PI_CENTROIDS_3BIT[idx] * PI_CENTROIDS_3BIT[idx];
    }

    float recon_norm = sqrtf(recon_sq);
    blk->norm = __float2half(recon_norm > 1e-10f ? grp_norm / recon_norm : grp_norm);
}

// ── Encode: Planar4 F32 → block_planar4_0 ───────────────────────────

__global__ void kernel_cpy_f32_planar4(
    const float * __restrict__ src,
    block_planar4_0 * __restrict__ dst,
    int64_t n_blocks)
{
    const int64_t ib = blockIdx.x * blockDim.x + threadIdx.x;
    if (ib >= n_blocks) return;

    const float * s = src + ib * QK_PLANAR4;
    block_planar4_0 * blk = &dst[ib];

    float buf[128];
    float norm_sq = 0.0f;
    #pragma unroll
    for (int j = 0; j < QK_PLANAR4; j++) {
        buf[j] = s[j];
        norm_sq += buf[j] * buf[j];
    }
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    #pragma unroll
    for (int j = 0; j < QK_PLANAR4; j++) buf[j] *= inv_norm;

    float rotated[128];
    #pragma unroll
    for (int p = 0; p < 64; p++) {
        float c = PI_COS[p], sv = PI_SIN[p];
        rotated[p*2]   = c * buf[p*2] - sv * buf[p*2+1];
        rotated[p*2+1] = sv * buf[p*2] + c * buf[p*2+1];
    }

    #pragma unroll
    for (int j = 0; j < 64; j++) blk->qs[j] = 0;
    float recon_sq = 0.0f;
    #pragma unroll
    for (int j = 0; j < 128; j++) {
        uint8_t idx = quantize_4bit(rotated[j]);
        blk->qs[j/2] |= (idx & 0xF) << ((j%2)*4);
        recon_sq += PI_CENTROIDS_4BIT[idx] * PI_CENTROIDS_4BIT[idx];
    }

    float recon_norm = sqrtf(recon_sq);
    blk->norm  = __float2half(recon_norm > 1e-10f ? grp_norm / recon_norm : grp_norm);
    blk->rnorm = __float2half(0.0f);
}

// ── Encode: Iso3 F32 → block_iso3_0 ─────────────────────────────────

__global__ void kernel_cpy_f32_iso3(
    const float * __restrict__ src,
    block_iso3_0 * __restrict__ dst,
    int64_t n_blocks)
{
    const int64_t ib = blockIdx.x * blockDim.x + threadIdx.x;
    if (ib >= n_blocks) return;

    const float * s = src + ib * QK_ISO3;
    block_iso3_0 * blk = &dst[ib];

    float buf[128];
    float norm_sq = 0.0f;
    #pragma unroll
    for (int j = 0; j < QK_ISO3; j++) {
        buf[j] = s[j];
        norm_sq += buf[j] * buf[j];
    }
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    #pragma unroll
    for (int j = 0; j < QK_ISO3; j++) buf[j] *= inv_norm;

    float rotated[128];
    #pragma unroll
    for (int g = 0; g < 32; g++) {
        float qw = PI_QW[g], qx = PI_QX[g], qy = PI_QY[g], qz = PI_QZ[g];
        float v0 = buf[g*4], v1 = buf[g*4+1], v2 = buf[g*4+2], v3 = buf[g*4+3];
        rotated[g*4]   = qw*v0 - qx*v1 - qy*v2 - qz*v3;
        rotated[g*4+1] = qw*v1 + qx*v0 + qy*v3 - qz*v2;
        rotated[g*4+2] = qw*v2 - qx*v3 + qy*v0 + qz*v1;
        rotated[g*4+3] = qw*v3 + qx*v2 - qy*v1 + qz*v0;
    }

    #pragma unroll
    for (int j = 0; j < QK_ISO3/4; j++) blk->qs[j] = 0;
    #pragma unroll
    for (int j = 0; j < QK_ISO3/8; j++) blk->signs[j] = 0;

    float recon_sq = 0.0f;
    #pragma unroll
    for (int j = 0; j < QK_ISO3; j++) {
        uint8_t idx = quantize_3bit(rotated[j]);
        blk->qs[j/4] |= (idx & 0x3) << ((j%4)*2);
        if (idx & 0x4) blk->signs[j/8] |= (1 << (j%8));
        recon_sq += PI_CENTROIDS_3BIT[idx] * PI_CENTROIDS_3BIT[idx];
    }

    float recon_norm = sqrtf(recon_sq);
    blk->norm = __float2half(recon_norm > 1e-10f ? grp_norm / recon_norm : grp_norm);
}

// ── Encode: Iso4 F32 → block_iso4_0 ─────────────────────────────────

__global__ void kernel_cpy_f32_iso4(
    const float * __restrict__ src,
    block_iso4_0 * __restrict__ dst,
    int64_t n_blocks)
{
    const int64_t ib = blockIdx.x * blockDim.x + threadIdx.x;
    if (ib >= n_blocks) return;

    const float * s = src + ib * QK_ISO4;
    block_iso4_0 * blk = &dst[ib];

    float buf[128];
    float norm_sq = 0.0f;
    #pragma unroll
    for (int j = 0; j < QK_ISO4; j++) {
        buf[j] = s[j];
        norm_sq += buf[j] * buf[j];
    }
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    #pragma unroll
    for (int j = 0; j < QK_ISO4; j++) buf[j] *= inv_norm;

    float rotated[128];
    #pragma unroll
    for (int g = 0; g < 32; g++) {
        float qw = PI_QW[g], qx = PI_QX[g], qy = PI_QY[g], qz = PI_QZ[g];
        float v0 = buf[g*4], v1 = buf[g*4+1], v2 = buf[g*4+2], v3 = buf[g*4+3];
        rotated[g*4]   = qw*v0 - qx*v1 - qy*v2 - qz*v3;
        rotated[g*4+1] = qw*v1 + qx*v0 + qy*v3 - qz*v2;
        rotated[g*4+2] = qw*v2 - qx*v3 + qy*v0 + qz*v1;
        rotated[g*4+3] = qw*v3 + qx*v2 - qy*v1 + qz*v0;
    }

    #pragma unroll
    for (int j = 0; j < 64; j++) blk->qs[j] = 0;
    float recon_sq = 0.0f;
    #pragma unroll
    for (int j = 0; j < 128; j++) {
        uint8_t idx = quantize_4bit(rotated[j]);
        blk->qs[j/2] |= (idx & 0xF) << ((j%2)*4);
        recon_sq += PI_CENTROIDS_4BIT[idx] * PI_CENTROIDS_4BIT[idx];
    }

    float recon_norm = sqrtf(recon_sq);
    blk->norm  = __float2half(recon_norm > 1e-10f ? grp_norm / recon_norm : grp_norm);
    blk->rnorm = __float2half(0.0f);
}

// ── Host dispatch (encode) ───────────────────────────────────────────

void ggml_cuda_cpy_f16_planar3(const char * src, char * dst, int64_t ne, cudaStream_t stream) {
    const int64_t n_blocks = ne / QK_PLANAR3;
    if (n_blocks == 0) return;
    const int threads = 256;
    const int blocks = (n_blocks + threads - 1) / threads;
    kernel_cpy_f16_planar3<<<blocks, threads, 0, stream>>>(
        (const half *)src, (block_planar3_0 *)dst, n_blocks);
}

void ggml_cuda_cpy_f16_planar4(const char * src, char * dst, int64_t ne, cudaStream_t stream) {
    const int64_t n_blocks = ne / QK_PLANAR4;
    if (n_blocks == 0) return;
    const int threads = 256;
    const int blocks = (n_blocks + threads - 1) / threads;
    kernel_cpy_f16_planar4<<<blocks, threads, 0, stream>>>(
        (const half *)src, (block_planar4_0 *)dst, n_blocks);
}

void ggml_cuda_cpy_f16_iso3(const char * src, char * dst, int64_t ne, cudaStream_t stream) {
    const int64_t n_blocks = ne / QK_ISO3;
    if (n_blocks == 0) return;
    const int threads = 256;
    const int blocks = (n_blocks + threads - 1) / threads;
    kernel_cpy_f16_iso3<<<blocks, threads, 0, stream>>>(
        (const half *)src, (block_iso3_0 *)dst, n_blocks);
}

void ggml_cuda_cpy_f16_iso4(const char * src, char * dst, int64_t ne, cudaStream_t stream) {
    const int64_t n_blocks = ne / QK_ISO4;
    if (n_blocks == 0) return;
    const int threads = 256;
    const int blocks = (n_blocks + threads - 1) / threads;
    kernel_cpy_f16_iso4<<<blocks, threads, 0, stream>>>(
        (const half *)src, (block_iso4_0 *)dst, n_blocks);
}

void ggml_cuda_cpy_f32_planar3(const char * src, char * dst, int64_t ne, cudaStream_t stream) {
    const int64_t n_blocks = ne / QK_PLANAR3;
    if (n_blocks == 0) return;
    const int threads = 256;
    const int blocks = (n_blocks + threads - 1) / threads;
    kernel_cpy_f32_planar3<<<blocks, threads, 0, stream>>>(
        (const float *)src, (block_planar3_0 *)dst, n_blocks);
}

void ggml_cuda_cpy_f32_planar4(const char * src, char * dst, int64_t ne, cudaStream_t stream) {
    const int64_t n_blocks = ne / QK_PLANAR4;
    if (n_blocks == 0) return;
    const int threads = 256;
    const int blocks = (n_blocks + threads - 1) / threads;
    kernel_cpy_f32_planar4<<<blocks, threads, 0, stream>>>(
        (const float *)src, (block_planar4_0 *)dst, n_blocks);
}

void ggml_cuda_cpy_f32_iso3(const char * src, char * dst, int64_t ne, cudaStream_t stream) {
    const int64_t n_blocks = ne / QK_ISO3;
    if (n_blocks == 0) return;
    const int threads = 256;
    const int blocks = (n_blocks + threads - 1) / threads;
    kernel_cpy_f32_iso3<<<blocks, threads, 0, stream>>>(
        (const float *)src, (block_iso3_0 *)dst, n_blocks);
}

void ggml_cuda_cpy_f32_iso4(const char * src, char * dst, int64_t ne, cudaStream_t stream) {
    const int64_t n_blocks = ne / QK_ISO4;
    if (n_blocks == 0) return;
    const int threads = 256;
    const int blocks = (n_blocks + threads - 1) / threads;
    kernel_cpy_f32_iso4<<<blocks, threads, 0, stream>>>(
        (const float *)src, (block_iso4_0 *)dst, n_blocks);
}

// ── Decode: planar3/4, iso3/4 → F16 (for fattn-mma path) ────────────

__global__ void kernel_dequant_planar3_f16(
    const block_planar3_0 * __restrict__ src,
    half * __restrict__ dst,
    int64_t n_blocks)
{
    const int64_t ib = blockIdx.x * blockDim.x + threadIdx.x;
    if (ib >= n_blocks) return;

    const block_planar3_0 * blk = &src[ib];
    float norm = __half2float(blk->norm);

    /* Inverse Givens rotation: R^T = [c s; -s c] */
    #pragma unroll
    for (int p = 0; p < 64; p++) {
        int j0 = p*2, j1 = p*2+1;
        uint8_t low0 = (blk->qs[j0/4] >> ((j0%4)*2)) & 0x3;
        uint8_t hi0  = (blk->signs[j0/8] >> (j0%8)) & 0x1;
        uint8_t low1 = (blk->qs[j1/4] >> ((j1%4)*2)) & 0x3;
        uint8_t hi1  = (blk->signs[j1/8] >> (j1%8)) & 0x1;
        float q0 = PI_CENTROIDS_3BIT[low0 | (hi0 << 2)];
        float q1 = PI_CENTROIDS_3BIT[low1 | (hi1 << 2)];
        float c = PI_COS[p], s = PI_SIN[p];
        dst[ib * QK_PLANAR3 + j0] = __float2half((c * q0 + s * q1) * norm);
        dst[ib * QK_PLANAR3 + j1] = __float2half((-s * q0 + c * q1) * norm);
    }
}

__global__ void kernel_dequant_planar4_f16(
    const block_planar4_0 * __restrict__ src,
    half * __restrict__ dst,
    int64_t n_blocks)
{
    const int64_t ib = blockIdx.x * blockDim.x + threadIdx.x;
    if (ib >= n_blocks) return;

    const block_planar4_0 * blk = &src[ib];
    float norm = __half2float(blk->norm);

    #pragma unroll
    for (int p = 0; p < 64; p++) {
        int j0 = p*2, j1 = p*2+1;
        uint8_t i0 = (blk->qs[j0/2] >> ((j0%2)*4)) & 0xF;
        uint8_t i1 = (blk->qs[j1/2] >> ((j1%2)*4)) & 0xF;
        float q0 = PI_CENTROIDS_4BIT[i0];
        float q1 = PI_CENTROIDS_4BIT[i1];
        float c = PI_COS[p], s = PI_SIN[p];
        dst[ib * QK_PLANAR4 + j0] = __float2half((c * q0 + s * q1) * norm);
        dst[ib * QK_PLANAR4 + j1] = __float2half((-s * q0 + c * q1) * norm);
    }
}

__global__ void kernel_dequant_iso3_f16(
    const block_iso3_0 * __restrict__ src,
    half * __restrict__ dst,
    int64_t n_blocks)
{
    const int64_t ib = blockIdx.x * blockDim.x + threadIdx.x;
    if (ib >= n_blocks) return;

    const block_iso3_0 * blk = &src[ib];
    float norm = __half2float(blk->norm);

    /* Inverse rotation: conj(q_L) * v where conj(q) = (qw, -qx, -qy, -qz) */
    #pragma unroll
    for (int g = 0; g < 32; g++) {
        float qvals[4];
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            int j = g*4 + c;
            uint8_t low = (blk->qs[j/4] >> ((j%4)*2)) & 0x3;
            uint8_t hi  = (blk->signs[j/8] >> (j%8)) & 0x1;
            qvals[c] = PI_CENTROIDS_3BIT[low | (hi << 2)];
        }
        float qw = PI_QW[g], qx = -PI_QX[g], qy = -PI_QY[g], qz = -PI_QZ[g];
        float v0 = qvals[0], v1 = qvals[1], v2 = qvals[2], v3 = qvals[3];
        dst[ib*QK_ISO3 + g*4+0] = __float2half((qw*v0 - qx*v1 - qy*v2 - qz*v3) * norm);
        dst[ib*QK_ISO3 + g*4+1] = __float2half((qw*v1 + qx*v0 + qy*v3 - qz*v2) * norm);
        dst[ib*QK_ISO3 + g*4+2] = __float2half((qw*v2 - qx*v3 + qy*v0 + qz*v1) * norm);
        dst[ib*QK_ISO3 + g*4+3] = __float2half((qw*v3 + qx*v2 - qy*v1 + qz*v0) * norm);
    }
}

__global__ void kernel_dequant_iso4_f16(
    const block_iso4_0 * __restrict__ src,
    half * __restrict__ dst,
    int64_t n_blocks)
{
    const int64_t ib = blockIdx.x * blockDim.x + threadIdx.x;
    if (ib >= n_blocks) return;

    const block_iso4_0 * blk = &src[ib];
    float norm = __half2float(blk->norm);

    #pragma unroll
    for (int g = 0; g < 32; g++) {
        float qvals[4];
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            int j = g*4 + c;
            uint8_t idx = (blk->qs[j/2] >> ((j%2)*4)) & 0xF;
            qvals[c] = PI_CENTROIDS_4BIT[idx];
        }
        float qw = PI_QW[g], qx = -PI_QX[g], qy = -PI_QY[g], qz = -PI_QZ[g];
        float v0 = qvals[0], v1 = qvals[1], v2 = qvals[2], v3 = qvals[3];
        dst[ib*QK_ISO4 + g*4+0] = __float2half((qw*v0 - qx*v1 - qy*v2 - qz*v3) * norm);
        dst[ib*QK_ISO4 + g*4+1] = __float2half((qw*v1 + qx*v0 + qy*v3 - qz*v2) * norm);
        dst[ib*QK_ISO4 + g*4+2] = __float2half((qw*v2 - qx*v3 + qy*v0 + qz*v1) * norm);
        dst[ib*QK_ISO4 + g*4+3] = __float2half((qw*v3 + qx*v2 - qy*v1 + qz*v0) * norm);
    }
}

// ── Host dispatch (decode, signature matches to_fp16_cuda_t) ─────────

void dequantize_row_planar3_0_cuda(const void * __restrict__ x, half * __restrict__ y,
                                   int64_t nrows, int64_t n_per_row, cudaStream_t stream) {
    const int64_t n_blocks = (nrows * n_per_row) / QK_PLANAR3;
    if (n_blocks == 0) return;
    const int threads = 256;
    const int blocks = (n_blocks + threads - 1) / threads;
    kernel_dequant_planar3_f16<<<blocks, threads, 0, stream>>>(
        (const block_planar3_0 *)x, y, n_blocks);
}

void dequantize_row_planar4_0_cuda(const void * __restrict__ x, half * __restrict__ y,
                                   int64_t nrows, int64_t n_per_row, cudaStream_t stream) {
    const int64_t n_blocks = (nrows * n_per_row) / QK_PLANAR4;
    if (n_blocks == 0) return;
    const int threads = 256;
    const int blocks = (n_blocks + threads - 1) / threads;
    kernel_dequant_planar4_f16<<<blocks, threads, 0, stream>>>(
        (const block_planar4_0 *)x, y, n_blocks);
}

void dequantize_row_iso3_0_cuda(const void * __restrict__ x, half * __restrict__ y,
                                int64_t nrows, int64_t n_per_row, cudaStream_t stream) {
    const int64_t n_blocks = (nrows * n_per_row) / QK_ISO3;
    if (n_blocks == 0) return;
    const int threads = 256;
    const int blocks = (n_blocks + threads - 1) / threads;
    kernel_dequant_iso3_f16<<<blocks, threads, 0, stream>>>(
        (const block_iso3_0 *)x, y, n_blocks);
}

void dequantize_row_iso4_0_cuda(const void * __restrict__ x, half * __restrict__ y,
                                int64_t nrows, int64_t n_per_row, cudaStream_t stream) {
    const int64_t n_blocks = (nrows * n_per_row) / QK_ISO4;
    if (n_blocks == 0) return;
    const int threads = 256;
    const int blocks = (n_blocks + threads - 1) / threads;
    kernel_dequant_iso4_f16<<<blocks, threads, 0, stream>>>(
        (const block_iso4_0 *)x, y, n_blocks);
}

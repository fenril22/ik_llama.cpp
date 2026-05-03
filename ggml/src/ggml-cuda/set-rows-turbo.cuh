#pragma once
/*
 * TurboQuant set_rows device functions.
 * Called via k_set_rows_quant template: 1 thread processes QK elements serially.
 * Input is already WHT-rotated (by ggml_turbo_wht op in the graph).
 * Normalization: L2 norm — matches CPU path and Lloyd-Max centroid assumption.
 */

#include "ggml-common.h"
#include "turbo-quant.cuh"
#include <cuda_fp16.h>

// turbo3: L2 norm + 3-bit centroid pack (128 elements, serial)
__device__ __forceinline__ void quantize_f32_turbo3_0_setrows(
        const float * __restrict__ src, block_turbo3_0 * __restrict__ dst) {

    // 1. L2 norm
    float norm_sq = 0.0f;
    for (int j = 0; j < QK_TURBO3; j++) {
        norm_sq += src[j] * src[j];
    }
    const float grp_norm = sqrtf(norm_sq);
    const float inv = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    // 2. Quantize + pack
    for (int j = 0; j < QK_TURBO3 / 4; j++) dst->qs[j] = 0;
    for (int j = 0; j < QK_TURBO3 / 8; j++) dst->signs[j] = 0;

    float recon_sq = 0.0f;
    for (int j = 0; j < QK_TURBO3; j++) {
        const uint8_t idx = turbo_nearest_centroid_3bit(src[j] * inv);
        dst->qs[j / 4]    |= (idx & 0x3) << ((j % 4) * 2);
        if (idx & 0x4) dst->signs[j / 8] |= (1 << (j % 8));
        const float c = TURBO_CENTROIDS_3BIT[idx];
        recon_sq += c * c;
    }

    // 3. Corrected norm
    const float rn = sqrtf(recon_sq);
    dst->norm = __float2half((rn > 1e-10f) ? grp_norm / rn : grp_norm);
}

// turbo4: L2 norm + 4-bit centroid pack (128 elements, serial)
__device__ __forceinline__ void quantize_f32_turbo4_0_setrows(
        const float * __restrict__ src, block_turbo4_0 * __restrict__ dst) {

    float norm_sq = 0.0f;
    for (int j = 0; j < QK_TURBO4; j++) {
        norm_sq += src[j] * src[j];
    }
    const float grp_norm = sqrtf(norm_sq);
    const float inv = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    for (int j = 0; j < QK_TURBO4 / 2; j++) dst->qs[j] = 0;

    float recon_sq = 0.0f;
    for (int j = 0; j < QK_TURBO4; j++) {
        const uint8_t idx = turbo_nearest_centroid_4bit(src[j] * inv);
        dst->qs[j / 2] |= (idx & 0xF) << ((j % 2) * 4);
        const float c = TURBO_CENTROIDS_4BIT[idx];
        recon_sq += c * c;
    }

    const float rn = sqrtf(recon_sq);
    dst->norm  = __float2half((rn > 1e-10f) ? grp_norm / rn : grp_norm);
    dst->rnorm = __float2half(0.0f);
}

// turbo3c: L2 norm + 3-bit integer-table pack (128 elements, serial, DP4A compatible)
__device__ __forceinline__ void quantize_f32_turbo3c_0_setrows(
        const float * __restrict__ src, block_turbo3c_0 * __restrict__ dst) {

    float norm_sq = 0.0f;
    for (int j = 0; j < QK_TURBO3C; j++) {
        norm_sq += src[j] * src[j];
    }
    const float grp_norm = sqrtf(norm_sq);
    const float inv = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    for (int j = 0; j < QK_TURBO3C / 4; j++) dst->qs[j] = 0;
    for (int j = 0; j < QK_TURBO3C / 8; j++) dst->signs[j] = 0;

    float recon_sq = 0.0f;
    for (int j = 0; j < QK_TURBO3C; j++) {
        const uint8_t idx = turbo3c_nearest_centroid(src[j] * inv);
        dst->qs[j / 4]    |= (idx & 0x3) << ((j % 4) * 2);
        if (idx & 0x4) dst->signs[j / 8] |= (1 << (j % 8));
        const float c = TURBO_CENTROIDS_3CBIT[idx];
        recon_sq += c * c;
    }

    const float rn = sqrtf(recon_sq);
    dst->norm = __float2half((rn > 1e-10f) ? grp_norm / rn : grp_norm);
}

// turbo2: L2 norm + 2-bit centroid pack (128 elements, serial)
__device__ __forceinline__ void quantize_f32_turbo2_0_setrows(
        const float * __restrict__ src, block_turbo2_0 * __restrict__ dst) {

    float norm_sq = 0.0f;
    for (int j = 0; j < QK_TURBO2; j++) {
        norm_sq += src[j] * src[j];
    }
    const float grp_norm = sqrtf(norm_sq);
    const float inv = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    for (int j = 0; j < QK_TURBO2 / 4; j++) dst->qs[j] = 0;

    float recon_sq = 0.0f;
    for (int j = 0; j < QK_TURBO2; j++) {
        const uint8_t idx = turbo_nearest_centroid_2bit(src[j] * inv);
        dst->qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
        const float c = TURBO_CENTROIDS_2BIT[idx];
        recon_sq += c * c;
    }

    const float rn = sqrtf(recon_sq);
    dst->norm = __float2half((rn > 1e-10f) ? grp_norm / rn : grp_norm);
}

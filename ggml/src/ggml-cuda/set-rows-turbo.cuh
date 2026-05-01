#pragma once
// TurboQuant set_rows device functions.
// Called AFTER WHT has been applied in the graph (Q=forward WHT, KV=forward WHT before set_rows).
// So quantize_func only needs to do absmax norm + centroid lookup, no rotation.

#include "ggml-common.h"
#include "turbo-quant.cuh"
#include <cuda_fp16.h>

// turbo3: absmax norm + 3-bit centroid pack
__device__ __forceinline__ void quantize_f32_turbo3_0_setrows(
        const float * __restrict__ src, block_turbo3_0 * __restrict__ dst) {

    // 1. absmax normalization
    float grp_norm = 0.0f;
    for (int j = 0; j < QK_TURBO3; j++) {
        float av = fabsf(src[j]);
        if (av > grp_norm) grp_norm = av;
    }
    const float inv = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    // 2. Quantize
    for (int j = 0; j < QK_TURBO3 / 4; j++) dst->qs[j] = 0;
    for (int j = 0; j < QK_TURBO3 / 8; j++) dst->signs[j] = 0;

    for (int j = 0; j < QK_TURBO3; j++) {
        uint8_t idx = turbo_nearest_centroid_3bit(src[j] * inv);
        dst->qs[j / 4]    |= (idx & 0x3) << ((j % 4) * 2);
        if (idx & 0x4) dst->signs[j / 8] |= (1 << (j % 8));
    }

    // 3. Norm: recon_norm correction (mirror CPU path)
    float recon_sq = 0.0f;
    for (int j = 0; j < QK_TURBO3; j++) {
        uint8_t low2 = (dst->qs[j / 4] >> ((j % 4) * 2)) & 0x3;
        uint8_t hi1  = (dst->signs[j / 8] >> (j % 8)) & 0x1;
        float c = TURBO_CENTROIDS_3BIT[low2 | (hi1 << 2)];
        recon_sq += c * c;
    }
    float rn = sqrtf(recon_sq);
    dst->norm = __float2half((rn > 1e-10f) ? grp_norm / rn : grp_norm);
}

// turbo4: absmax norm + 4-bit centroid pack
__device__ __forceinline__ void quantize_f32_turbo4_0_setrows(
        const float * __restrict__ src, block_turbo4_0 * __restrict__ dst) {

    float grp_norm = 0.0f;
    for (int j = 0; j < QK_TURBO4; j++) {
        float av = fabsf(src[j]);
        if (av > grp_norm) grp_norm = av;
    }
    const float inv = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    for (int j = 0; j < QK_TURBO4 / 2; j++) dst->qs[j] = 0;

    for (int j = 0; j < QK_TURBO4; j++) {
        uint8_t idx = turbo_nearest_centroid_4bit(src[j] * inv);
        dst->qs[j / 2] |= (idx & 0xF) << ((j % 2) * 4);
    }

    float recon_sq = 0.0f;
    for (int j = 0; j < QK_TURBO4; j++) {
        float c = TURBO_CENTROIDS_4BIT[(dst->qs[j/2] >> ((j%2)*4)) & 0xF];
        recon_sq += c * c;
    }
    float rn = sqrtf(recon_sq);
    dst->norm  = __float2half((rn > 1e-10f) ? grp_norm / rn : grp_norm);
    dst->rnorm = __float2half(0.0f);
}

// turbo2: absmax norm + 2-bit centroid pack
__device__ __forceinline__ void quantize_f32_turbo2_0_setrows(
        const float * __restrict__ src, block_turbo2_0 * __restrict__ dst) {

    float grp_norm = 0.0f;
    for (int j = 0; j < QK_TURBO2; j++) {
        float av = fabsf(src[j]);
        if (av > grp_norm) grp_norm = av;
    }
    const float inv = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    for (int j = 0; j < QK_TURBO2 / 4; j++) dst->qs[j] = 0;

    for (int j = 0; j < QK_TURBO2; j++) {
        uint8_t idx = turbo_nearest_centroid_2bit(src[j] * inv);
        dst->qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
    }

    float recon_sq = 0.0f;
    for (int j = 0; j < QK_TURBO2; j++) {
        float c = TURBO_CENTROIDS_2BIT[(dst->qs[j/4] >> ((j%4)*2)) & 0x3];
        recon_sq += c * c;
    }
    float rn = sqrtf(recon_sq);
    dst->norm = __float2half((rn > 1e-10f) ? grp_norm / rn : grp_norm);
}

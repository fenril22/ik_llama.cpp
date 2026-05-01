#pragma once

// Device quantize functions for planar3/iso3/planar4/iso4 set_rows.
// These wrap the rotation + quantize logic into the __device__ function
// signature required by k_set_rows_quant.
//
// Uses static __constant__ arrays from planar-iso-constants.cuh so each
// compilation unit gets its own initialized copy — no cross-TU extern needed.

#include "ggml-common.h"
#include "planar-iso-constants.cuh"
#include <cuda_fp16.h>
#include <cmath>

// ── Helpers ─────────────────────────────────────────────────────────

__device__ __forceinline__ uint8_t sr_quantize_3bit(float val, const float * mid) {
    if      (val < mid[0]) return 0;
    else if (val < mid[1]) return 1;
    else if (val < mid[2]) return 2;
    else if (val < mid[3]) return 3;
    else if (val < mid[4]) return 4;
    else if (val < mid[5]) return 5;
    else if (val < mid[6]) return 6;
    else                   return 7;
}

__device__ __forceinline__ uint8_t sr_quantize_4bit(float val, const float * /*unused*/) {
    // O(1) midpoint comparison; PI_MID_4BIT from planar-iso-constants.cuh
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

// ── Planar3: F32[128] → block_planar3_0 ─────────────────────────────

__device__ void quantize_f32_planar3_block(const float * x, block_planar3_0 * dst) {
    // Norm
    float norm_sq = 0.0f;
    float buf[128];
    for (int j = 0; j < QK_PLANAR3; j++) {
        buf[j] = x[j];
        norm_sq += buf[j] * buf[j];
    }
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    for (int j = 0; j < QK_PLANAR3; j++) buf[j] *= inv_norm;

    // Forward Givens rotation
    float rotated[128];
    for (int p = 0; p < 64; p++) {
        float c = PI_COS[p], s = PI_SIN[p];
        rotated[p*2]   = c * buf[p*2] - s * buf[p*2+1];
        rotated[p*2+1] = s * buf[p*2] + c * buf[p*2+1];
    }

    // Quantize + pack
    for (int j = 0; j < QK_PLANAR3/4; j++) dst->qs[j] = 0;
    for (int j = 0; j < QK_PLANAR3/8; j++) dst->signs[j] = 0;

    float recon_sq = 0.0f;
    for (int j = 0; j < QK_PLANAR3; j++) {
        uint8_t idx = sr_quantize_3bit(rotated[j], PI_MID_3BIT);
        dst->qs[j/4] |= (idx & 0x3) << ((j%4)*2);
        if (idx & 0x4) dst->signs[j/8] |= (1 << (j%8));
        recon_sq += PI_CENTROIDS_3BIT[idx] * PI_CENTROIDS_3BIT[idx];
    }

    float recon_norm = sqrtf(recon_sq);
    dst->norm = __float2half(recon_norm > 1e-10f ? grp_norm / recon_norm : grp_norm);
}

// ── Iso3: F32[128] → block_iso3_0 (quaternion rotation) ────────────

__device__ void quantize_f32_iso3_block(const float * x, block_iso3_0 * dst) {
    float norm_sq = 0.0f;
    float buf[128];
    for (int j = 0; j < QK_ISO3; j++) {
        buf[j] = x[j];
        norm_sq += buf[j] * buf[j];
    }
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    for (int j = 0; j < QK_ISO3; j++) buf[j] *= inv_norm;

    // Forward quaternion rotation per 4D group
    float rotated[128];
    for (int g = 0; g < 32; g++) {
        float qw = PI_QW[g], qx = PI_QX[g], qy = PI_QY[g], qz = PI_QZ[g];
        float v0 = buf[g*4], v1 = buf[g*4+1], v2 = buf[g*4+2], v3 = buf[g*4+3];
        rotated[g*4]   = qw*v0 - qx*v1 - qy*v2 - qz*v3;
        rotated[g*4+1] = qw*v1 + qx*v0 + qy*v3 - qz*v2;
        rotated[g*4+2] = qw*v2 - qx*v3 + qy*v0 + qz*v1;
        rotated[g*4+3] = qw*v3 + qx*v2 - qy*v1 + qz*v0;
    }

    for (int j = 0; j < QK_ISO3/4; j++) dst->qs[j] = 0;
    for (int j = 0; j < QK_ISO3/8; j++) dst->signs[j] = 0;

    float recon_sq = 0.0f;
    for (int j = 0; j < QK_ISO3; j++) {
        uint8_t idx = sr_quantize_3bit(rotated[j], PI_MID_3BIT);
        dst->qs[j/4] |= (idx & 0x3) << ((j%4)*2);
        if (idx & 0x4) dst->signs[j/8] |= (1 << (j%8));
        recon_sq += PI_CENTROIDS_3BIT[idx] * PI_CENTROIDS_3BIT[idx];
    }

    float recon_norm = sqrtf(recon_sq);
    dst->norm = __float2half(recon_norm > 1e-10f ? grp_norm / recon_norm : grp_norm);
}

// ── Planar4: F32[128] → block_planar4_0 (Givens + 4-bit nibble) ────

__device__ void quantize_f32_planar4_block(const float * x, block_planar4_0 * dst) {
    float norm_sq = 0.0f;
    float buf[128];
    for (int j = 0; j < QK_PLANAR4; j++) {
        buf[j] = x[j];
        norm_sq += buf[j] * buf[j];
    }
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    for (int j = 0; j < QK_PLANAR4; j++) buf[j] *= inv_norm;

    float rotated[128];
    for (int p = 0; p < 64; p++) {
        float c = PI_COS[p], s = PI_SIN[p];
        rotated[p*2]   = c * buf[p*2] - s * buf[p*2+1];
        rotated[p*2+1] = s * buf[p*2] + c * buf[p*2+1];
    }

    for (int j = 0; j < 64; j++) dst->qs[j] = 0;
    float recon_sq = 0.0f;
    for (int j = 0; j < 128; j++) {
        uint8_t idx = sr_quantize_4bit(rotated[j], PI_CENTROIDS_4BIT);
        dst->qs[j/2] |= (idx & 0xF) << ((j%2)*4);
        recon_sq += PI_CENTROIDS_4BIT[idx] * PI_CENTROIDS_4BIT[idx];
    }

    float recon_norm = sqrtf(recon_sq);
    dst->norm = __float2half(recon_norm > 1e-10f ? grp_norm / recon_norm : grp_norm);
    dst->rnorm = __float2half(0.0f);
}

// ── Iso4: F32[128] → block_iso4_0 (quaternion + 4-bit nibble) ──────

__device__ void quantize_f32_iso4_block(const float * x, block_iso4_0 * dst) {
    float norm_sq = 0.0f;
    float buf[128];
    for (int j = 0; j < QK_ISO4; j++) {
        buf[j] = x[j];
        norm_sq += buf[j] * buf[j];
    }
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    for (int j = 0; j < QK_ISO4; j++) buf[j] *= inv_norm;

    float rotated[128];
    for (int g = 0; g < 32; g++) {
        float qw = PI_QW[g], qx = PI_QX[g], qy = PI_QY[g], qz = PI_QZ[g];
        float v0 = buf[g*4], v1 = buf[g*4+1], v2 = buf[g*4+2], v3 = buf[g*4+3];
        rotated[g*4]   = qw*v0 - qx*v1 - qy*v2 - qz*v3;
        rotated[g*4+1] = qw*v1 + qx*v0 + qy*v3 - qz*v2;
        rotated[g*4+2] = qw*v2 - qx*v3 + qy*v0 + qz*v1;
        rotated[g*4+3] = qw*v3 + qx*v2 - qy*v1 + qz*v0;
    }

    for (int j = 0; j < 64; j++) dst->qs[j] = 0;
    float recon_sq = 0.0f;
    for (int j = 0; j < 128; j++) {
        uint8_t idx = sr_quantize_4bit(rotated[j], PI_CENTROIDS_4BIT);
        dst->qs[j/2] |= (idx & 0xF) << ((j%2)*4);
        recon_sq += PI_CENTROIDS_4BIT[idx] * PI_CENTROIDS_4BIT[idx];
    }

    float recon_norm = sqrtf(recon_sq);
    dst->norm = __float2half(recon_norm > 1e-10f ? grp_norm / recon_norm : grp_norm);
    dst->rnorm = __float2half(0.0f);
}

// ══════════════════════════════════════════════════════════════════════
// V-cache variants: NO ROTATION (for transposed V cache)
// ══════════════════════════════════════════════════════════════════════

__device__ void quantize_f32_planar3_block_norot(const float * x, block_planar3_0 * dst) {
    float norm_sq = 0.0f;
    float buf[128];
    for (int j = 0; j < QK_PLANAR3; j++) { buf[j] = x[j]; norm_sq += buf[j]*buf[j]; }
    float grp_norm = sqrtf(norm_sq);
    float inv = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    for (int j = 0; j < QK_PLANAR3; j++) buf[j] *= inv;
    for (int j = 0; j < QK_PLANAR3/4; j++) dst->qs[j] = 0;
    for (int j = 0; j < QK_PLANAR3/8; j++) dst->signs[j] = 0;
    float recon_sq = 0.0f;
    for (int j = 0; j < QK_PLANAR3; j++) {
        uint8_t idx = sr_quantize_3bit(buf[j], PI_MID_3BIT);
        dst->qs[j/4] |= (idx & 0x3) << ((j%4)*2);
        if (idx & 0x4) dst->signs[j/8] |= (1 << (j%8));
        recon_sq += PI_CENTROIDS_3BIT[idx] * PI_CENTROIDS_3BIT[idx];
    }
    float rn = sqrtf(recon_sq);
    dst->norm = __float2half(rn > 1e-10f ? grp_norm / rn : grp_norm);
}

__device__ void quantize_f32_iso3_block_norot(const float * x, block_iso3_0 * dst) {
    quantize_f32_planar3_block_norot(x, (block_planar3_0 *)dst);
}

__device__ void quantize_f32_planar4_block_norot(const float * x, block_planar4_0 * dst) {
    float norm_sq = 0.0f;
    float buf[128];
    for (int j = 0; j < QK_PLANAR4; j++) { buf[j] = x[j]; norm_sq += buf[j]*buf[j]; }
    float grp_norm = sqrtf(norm_sq);
    float inv = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    for (int j = 0; j < QK_PLANAR4; j++) buf[j] *= inv;
    for (int j = 0; j < 64; j++) dst->qs[j] = 0;
    float recon_sq = 0.0f;
    for (int j = 0; j < 128; j++) {
        uint8_t idx = sr_quantize_4bit(buf[j], PI_CENTROIDS_4BIT);
        dst->qs[j/2] |= (idx & 0xF) << ((j%2)*4);
        recon_sq += PI_CENTROIDS_4BIT[idx] * PI_CENTROIDS_4BIT[idx];
    }
    float rn = sqrtf(recon_sq);
    dst->norm = __float2half(rn > 1e-10f ? grp_norm / rn : grp_norm);
    dst->rnorm = __float2half(0.0f);
}

__device__ void quantize_f32_iso4_block_norot(const float * x, block_iso4_0 * dst) {
    quantize_f32_planar4_block_norot(x, (block_planar4_0 *)dst);
}

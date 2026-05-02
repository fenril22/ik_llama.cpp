/*
 * TurboQuant CUDA kernels for KV cache compression
 * Based on: arXiv 2504.19874 (ICLR 2026)
 *
 * Implements GGML_TYPE_TURBO3_0 (3-bit PolarQuant, block size 32)
 * Constants, WHT rotation, quantize/dequantize device functions.
 */

#pragma once

#include "common.cuh"
#include "turbo-innerq.cuh"

// Logging: GGML_CUDA_LOG_* is only defined in ggml-cuda.cu (static function).
// This header may be included from other .cu/.cuh files, so use fprintf as fallback.
#ifndef GGML_CUDA_LOG_INFO
#  include <cstdio>
#  define GGML_CUDA_LOG_INFO(fmt, ...)  fprintf(stderr, "[turbo-quant] " fmt, ##__VA_ARGS__)
#  define GGML_CUDA_LOG_WARN(fmt, ...)  fprintf(stderr, "[turbo-quant WARN] " fmt, ##__VA_ARGS__)
#  define GGML_CUDA_LOG_ERROR(fmt, ...) fprintf(stderr, "[turbo-quant ERROR] " fmt, ##__VA_ARGS__)
#endif

// ---- Quantization ratios for dequantize_block template ----
#define QR_TURBO3 1  // Each dequantize call produces 2 consecutive elements (like q8_0)
#define QR_TURBO2 1  // Each dequantize call produces 2 consecutive elements (like q8_0)
#define QR_TURBO4 1  // Each dequantize call produces 2 consecutive elements (like q8_0)

// ---- 2-bit centroids (Lloyd-Max for N(0, 1/128)) ----

static __constant__ float TURBO_CENTROIDS_2BIT[4] = {
    -0.133462f, -0.039994f, 0.039994f, 0.133462f
};

static __constant__ float TURBO_MID_2BIT[3] = {
    -0.086728f, 0.0f, 0.086728f
};

// ---- 3-bit centroids (Lloyd-Max for N(0, 1/128)) ----

static __constant__ float TURBO_CENTROIDS_3BIT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

// ---- Midpoints for nearest centroid lookup ----

static __constant__ float TURBO_MID_3BIT[7] = {
    -0.154259f, -0.091775f, -0.043589f, 0.0f,
     0.043589f,  0.091775f,  0.154259f
};

// ---- WHT sign arrays (seed=42) ----

static __constant__ float TURBO_WHT_SIGNS1[128] = {
    -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
    -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f
};

static __constant__ float TURBO_WHT_SIGNS2[128] = {
    1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f
};

// ---- 64-element WHT sign arrays (first 64 of the 128-element arrays) ----

static __constant__ float TURBO_WHT_SIGNS1_64[64] = {
    -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f
};

static __constant__ float TURBO_WHT_SIGNS2_64[64] = {
    1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f
};

// ---- Fast Walsh-Hadamard Transform (in-place, normalized) ----
// O(n log n) = 896 ops for n=128

static __device__ __forceinline__ void turbo_fwht_128(float * x) {
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j];
                float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    const float inv_sqrt_128 = 0.08838834764831845f;
    for (int i = 0; i < 128; i++) {
        x[i] *= inv_sqrt_128;
    }
}

// ---- Fast Walsh-Hadamard Transform for 64-element groups ----
// O(n log n) = 384 ops for n=64

static __device__ __forceinline__ void turbo_fwht_64(float * x) {
    for (int h = 1; h < 64; h *= 2) {
        for (int i = 0; i < 64; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j];
                float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    const float inv_sqrt_64 = 0.125f;
    for (int i = 0; i < 64; i++) {
        x[i] *= inv_sqrt_64;
    }
}

// ---- Forward rotation: signs1 → FWHT → signs2 ----

static __device__ __forceinline__ void turbo_rotate_forward(float * x) {
    for (int i = 0; i < 128; i++) x[i] *= TURBO_WHT_SIGNS1[i];
    turbo_fwht_128(x);
    for (int i = 0; i < 128; i++) x[i] *= TURBO_WHT_SIGNS2[i];
}

// ---- Forward rotation for 64-element groups ----

static __device__ __forceinline__ void turbo_rotate_forward_64(float * x) {
    for (int i = 0; i < 64; i++) x[i] *= TURBO_WHT_SIGNS1_64[i];
    turbo_fwht_64(x);
    for (int i = 0; i < 64; i++) x[i] *= TURBO_WHT_SIGNS2_64[i];
}

// ---- InnerQ per-channel equalization ----
// Equalizes K channel variances before WHT rotation to reduce quantization error.
// Enabled via TURBO_INNERQ=N env var (N = calibration token count).
// Math: <Q/s, s*K> = <Q, K> preserves dot products.
// INNERQ_MAX_CHANNELS is defined in turbo-innerq.cuh
//
// Device variables are defined once in turbo-wht.cu to avoid multiple-definition
// errors when this header is included from multiple translation units.

extern __device__ float d_innerq_scale[INNERQ_MAX_CHANNELS];
extern __device__ float d_innerq_scale_inv[INNERQ_MAX_CHANNELS];
extern __device__ float d_innerq_sq_accum[INNERQ_MAX_CHANNELS];
extern __device__ int   d_innerq_count;
extern __device__ int   d_innerq_active;
extern __device__ int   d_innerq_calibrating;

// Host-side InnerQ state — defined once in turbo-wht.cu
extern int   innerq_enabled;
extern int   innerq_target_tokens;
extern float innerq_strength;
extern bool  innerq_initialized;

// Host-side InnerQ functions — defined once in turbo-wht.cu
void turbo_innerq_init(void);
void turbo_innerq_finalize(int group_size);
void turbo_innerq_check_finalize(int group_size, int64_t ne00);
bool turbo_innerq_is_active(void);

// ---- 4-bit centroids (Lloyd-Max for N(0, 1/128)) ----

static __constant__ float TURBO_CENTROIDS_4BIT[16] = {
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
     0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f
};

// ---- Midpoints for nearest 4-bit centroid lookup ----

static __constant__ float TURBO_MID_4BIT[15] = {
    -0.145561f, -0.103361f, -0.079142f, -0.060009f,
    -0.043430f, -0.028293f, -0.013964f,  0.000000f,
     0.013964f,  0.028293f,  0.043430f,  0.060009f,
     0.079142f,  0.103361f,  0.145561f
};

// ---- Nearest 4-bit centroid index ----

static __device__ __forceinline__ uint8_t turbo_nearest_centroid_4bit(float val) {
    if      (val < TURBO_MID_4BIT[ 0]) return  0;
    else if (val < TURBO_MID_4BIT[ 1]) return  1;
    else if (val < TURBO_MID_4BIT[ 2]) return  2;
    else if (val < TURBO_MID_4BIT[ 3]) return  3;
    else if (val < TURBO_MID_4BIT[ 4]) return  4;
    else if (val < TURBO_MID_4BIT[ 5]) return  5;
    else if (val < TURBO_MID_4BIT[ 6]) return  6;
    else if (val < TURBO_MID_4BIT[ 7]) return  7;
    else if (val < TURBO_MID_4BIT[ 8]) return  8;
    else if (val < TURBO_MID_4BIT[ 9]) return  9;
    else if (val < TURBO_MID_4BIT[10]) return 10;
    else if (val < TURBO_MID_4BIT[11]) return 11;
    else if (val < TURBO_MID_4BIT[12]) return 12;
    else if (val < TURBO_MID_4BIT[13]) return 13;
    else if (val < TURBO_MID_4BIT[14]) return 14;
    else                               return 15;
}

// ---- Per-block quantize for turbo4 (128 elements, expects already-rotated input) ----

static __device__ void quantize_f32_turbo4_0_block(const float * __restrict__ src,
                                                    block_turbo4_0 * __restrict__ dst) {
    for (int j = 0; j < QK_TURBO4 / 2; j++) dst->qs[j] = 0;

    for (int j = 0; j < QK_TURBO4; j++) {
        uint8_t idx = turbo_nearest_centroid_4bit(src[j]);
        dst->qs[j / 2] |= (idx & 0xF) << ((j % 2) * 4);
    }
}

// ---- Inline dequant helper: extract one float from turbo4 block ----

static __device__ __forceinline__ float turbo4_dequant_element(
        const block_turbo4_0 * __restrict__ x, int j, float norm) {
    uint8_t idx = (x->qs[j / 2] >> ((j % 2) * 4)) & 0xF;
    return TURBO_CENTROIDS_4BIT[idx] * norm;
}

// ---- Nearest 3-bit centroid index ----

static __device__ __forceinline__ uint8_t turbo_nearest_centroid_3bit(float val) {
    if      (val < TURBO_MID_3BIT[0]) return 0;
    else if (val < TURBO_MID_3BIT[1]) return 1;
    else if (val < TURBO_MID_3BIT[2]) return 2;
    else if (val < TURBO_MID_3BIT[3]) return 3;
    else if (val < TURBO_MID_3BIT[4]) return 4;
    else if (val < TURBO_MID_3BIT[5]) return 5;
    else if (val < TURBO_MID_3BIT[6]) return 6;
    else                              return 7;
}

// ---- Per-block quantize (32 elements, expects already-rotated input) ----
// Used by set_rows after group-level WHT rotation

static __device__ void quantize_f32_turbo3_0_block(const float * __restrict__ src,
                                                    block_turbo3_0 * __restrict__ dst) {
    for (int j = 0; j < QK_TURBO3 / 4; j++) dst->qs[j] = 0;
    for (int j = 0; j < QK_TURBO3 / 8; j++) dst->signs[j] = 0;

    for (int j = 0; j < QK_TURBO3; j++) {
        uint8_t idx = turbo_nearest_centroid_3bit(src[j]);
        dst->qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
        if (idx & 0x4) {
            dst->signs[j / 8] |= (1 << (j % 8));
        }
    }
}

// ---- Inline dequant helper: extract one float from turbo3 block ----

static __device__ __forceinline__ float turbo3_dequant_element(
        const block_turbo3_0 * __restrict__ x, int j, float norm) {
    uint8_t low2 = (x->qs[j / 4] >> ((j % 4) * 2)) & 0x3;
    uint8_t hi1  = (x->signs[j / 8] >> (j % 8)) & 0x1;
    uint8_t idx  = low2 | (hi1 << 2);
    return TURBO_CENTROIDS_3BIT[idx] * norm;
}

// ---- Nearest 2-bit centroid index ----

static __device__ __forceinline__ uint8_t turbo_nearest_centroid_2bit(float val) {
    if      (val < TURBO_MID_2BIT[0]) return 0;
    else if (val < TURBO_MID_2BIT[1]) return 1;
    else if (val < TURBO_MID_2BIT[2]) return 2;
    else                              return 3;
}

// ---- Per-block quantize for turbo2 (32 elements, expects already-rotated input) ----

static __device__ void quantize_f32_turbo2_0_block(const float * __restrict__ src,
                                                    block_turbo2_0 * __restrict__ dst) {
    for (int j = 0; j < QK_TURBO2 / 4; j++) dst->qs[j] = 0;

    for (int j = 0; j < QK_TURBO2; j++) {
        uint8_t idx = turbo_nearest_centroid_2bit(src[j]);
        dst->qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
    }
}

// ---- Inline dequant helper: extract one float from turbo2 block ----

static __device__ __forceinline__ float turbo2_dequant_element(
        const block_turbo2_0 * __restrict__ x, int j, float norm) {
    uint8_t idx = (x->qs[j / 4] >> ((j % 4) * 2)) & 0x3;
    return TURBO_CENTROIDS_2BIT[idx] * norm;
}

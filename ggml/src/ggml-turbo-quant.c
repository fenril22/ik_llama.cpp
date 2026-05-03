/*
 * TurboQuant: KV cache compression via PolarQuant + WHT rotation
 * Based on: arXiv 2504.19874 (ICLR 2026)
 *
 * Implements GGML_TYPE_TURBO2_0 (2-bit), GGML_TYPE_TURBO3_0 (3-bit),
 * GGML_TYPE_TURBO4_0 (4-bit), and GGML_TYPE_TURBO3C_0 (3-bit DP4A)
 * for use as --cache-type-k turboN in llama-server.
 *
 * CPU quantization pipeline (all variants):
 *   1. Compute L2 norm of the rotation group
 *   2. Normalize by L2 norm
 *   3. Apply WHT rotation: signs1 → butterfly → 1/sqrt(n) → signs2
 *   4. Quantize each element to nearest Lloyd-Max centroid
 *   5. Store corrected norm: grp_norm / recon_norm
 *
 * This CPU path is a reference implementation and correctness fallback.
 * In normal GPU-accelerated operation, quantization is performed on-device
 * by cpy-turbo.cu (input pre-rotated by ggml_turbo_wht graph op).
 */

#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

/* Global: WHT group size for CPU quantize path (set by CPU SET_ROWS handler before each call) */
int turbo3_cpu_wht_group_size = 0;

/* ---------- Lloyd-Max centroids (optimized for N(0, 1/d)) ---------- */

/* 2-bit: 4 centroids */
static const float CENTROIDS_2BIT[4] = { -0.133462f, -0.039994f, 0.039994f, 0.133462f };

/* 3-bit: 8 centroids */
static const float CENTROIDS_3BIT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

/* 3C-bit: integer-table [-8,-5,-3,-1,1,3,5,8] * scale, DP4A compatible.
 * Scale = 0.023568 (least-squares fit to CENTROIDS_3BIT).
 * Stored as float for CPU dequantize; GPU uses integer DP4A path. */
static const float CENTROIDS_3CBIT[8] = {
    -8.0f * 0.023568f, -5.0f * 0.023568f, -3.0f * 0.023568f, -1.0f * 0.023568f,
     1.0f * 0.023568f,  3.0f * 0.023568f,  5.0f * 0.023568f,  8.0f * 0.023568f
};
/* Midpoints for nearest-centroid lookup */
static const float MID_3CBIT[7] = {
    (-8.0f - 5.0f) * 0.5f * 0.023568f,  // -6.5 * scale
    (-5.0f - 3.0f) * 0.5f * 0.023568f,  // -4.0 * scale
    (-3.0f - 1.0f) * 0.5f * 0.023568f,  // -2.0 * scale
    0.0f,
    ( 1.0f + 3.0f) * 0.5f * 0.023568f,  //  2.0 * scale
    ( 3.0f + 5.0f) * 0.5f * 0.023568f,  //  4.0 * scale
    ( 5.0f + 8.0f) * 0.5f * 0.023568f,  //  6.5 * scale
};

/* 4-bit: 16 centroids */
static const float CENTROIDS_4BIT[16] = {
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
     0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f
};

/* ---------- nearest centroid ---------- */

static int nearest_centroid_2bit(float val) {
    if (val < -0.086728f) return 0;
    if (val <  0.000000f) return 1;
    if (val <  0.086728f) return 2;
    return 3;
}

static int nearest_centroid_3bit(float val) {
    if (val < -0.154259f) return 0;
    if (val < -0.091775f) return 1;
    if (val < -0.043589f) return 2;
    if (val <  0.000000f) return 3;
    if (val <  0.043589f) return 4;
    if (val <  0.091775f) return 5;
    if (val <  0.154259f) return 6;
    return 7;
}

/* 3C: integer table [-8,-5,-3,-1,1,3,5,8] * 0.023568 */
static int nearest_centroid_3cbit(float val) {
    if (val < MID_3CBIT[0]) return 0;
    if (val < MID_3CBIT[1]) return 1;
    if (val < MID_3CBIT[2]) return 2;
    if (val < MID_3CBIT[3]) return 3;
    if (val < MID_3CBIT[4]) return 4;
    if (val < MID_3CBIT[5]) return 5;
    if (val < MID_3CBIT[6]) return 6;
    return 7;
}

static int nearest_centroid_4bit(float val) {
    if (val < -0.145561f) return 0;
    if (val < -0.103361f) return 1;
    if (val < -0.079142f) return 2;
    if (val < -0.060009f) return 3;
    if (val < -0.043430f) return 4;
    if (val < -0.028293f) return 5;
    if (val < -0.013964f) return 6;
    if (val <  0.000000f) return 7;
    if (val <  0.013964f) return 8;
    if (val <  0.028293f) return 9;
    if (val <  0.043430f) return 10;
    if (val <  0.060009f) return 11;
    if (val <  0.079142f) return 12;
    if (val <  0.103361f) return 13;
    if (val <  0.145561f) return 14;
    return 15;
}

/* ---------- WHT sign arrays (seed=42, must match CUDA) ---------- */

static const float turbo_cpu_s1[128] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1
};

static const float turbo_cpu_s2[128] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1
};

/* ---------- CPU forward WHT (in-place, group_size elements) ---------- */

static void turbo_cpu_fwht(float * x, int group_size) {
    const float inv_sqrt = (group_size == 128) ? 0.08838834764831845f : 0.125f;

    for (int i = 0; i < group_size; i++) x[i] *= turbo_cpu_s1[i];

    for (int h = 1; h < group_size; h *= 2) {
        for (int i = 0; i < group_size; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }

    for (int i = 0; i < group_size; i++) x[i] *= inv_sqrt * turbo_cpu_s2[i];
}

/* ---------- TURBO3_0: 3-bit PolarQuant with WHT rotation ---------- */

void quantize_row_turbo3_0_ref(const float * GGML_RESTRICT x, block_turbo3_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO3 == 0);

    extern int turbo3_cpu_wht_group_size;
    int group_size = turbo3_cpu_wht_group_size;
    if (group_size != 64 && group_size != 128) {
        group_size = (k % 128 == 0) ? 128 : 64;
    }
    if (k % group_size != 0) group_size = (group_size == 128) ? 64 : 128;
    assert(k % group_size == 0);

    const int n_groups = k / group_size;
    const int blocks_per_group = group_size / QK_TURBO3;

    for (int g = 0; g < n_groups; g++) {
        const float * grp_src = x + g * group_size;
        block_turbo3_0 * grp_dst = y + g * blocks_per_group;

        /* 1. L2 norm over the group */
        float norm_sq = 0.0f;
        float buf[128];
        for (int j = 0; j < group_size; j++) {
            buf[j] = grp_src[j];
            norm_sq += buf[j] * buf[j];
        }
        float grp_norm = sqrtf(norm_sq);
        float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

        /* 2. Normalize */
        for (int j = 0; j < group_size; j++) buf[j] *= inv_norm;

        /* 3. Forward WHT rotation */
        turbo_cpu_fwht(buf, group_size);

        /* 4. Quantize + pack into sub-blocks, accumulate reconstruction norm */
        float recon_sq = 0.0f;
        for (int b = 0; b < blocks_per_group; b++) {
            block_turbo3_0 * blk = &grp_dst[b];
            const int off = b * QK_TURBO3;

            memset(blk->qs,    0, QK_TURBO3 / 4);
            memset(blk->signs, 0, QK_TURBO3 / 8);

            for (int j = 0; j < QK_TURBO3; j++) {
                int idx = nearest_centroid_3bit(buf[off + j]);
                blk->qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
                if (idx & 0x4) {
                    blk->signs[j / 8] |= (1 << (j % 8));
                }
                recon_sq += CENTROIDS_3BIT[idx] * CENTROIDS_3BIT[idx];
            }
        }

        /* 5. Corrected norm: grp_norm / recon_norm (same formula as CUDA kernel) */
        float recon_norm = sqrtf(recon_sq);
        float corrected = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        for (int b = 0; b < blocks_per_group; b++) {
            grp_dst[b].norm = GGML_FP32_TO_FP16(corrected);
        }
    }
}

void dequantize_row_turbo3_0(const block_turbo3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO3 == 0);
    const int nb = k / QK_TURBO3;
    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int j = 0; j < QK_TURBO3; j++) {
            uint8_t low2 = (x[block].qs[j / 4] >> ((j % 4) * 2)) & 0x3;
            uint8_t hi1  = (x[block].signs[j / 8] >> (j % 8)) & 0x1;
            uint8_t idx  = low2 | (hi1 << 2);
            y[block * QK_TURBO3 + j] = CENTROIDS_3BIT[idx] * norm;
        }
    }
}

size_t quantize_turbo3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO3 == 0);

    size_t row_size = (n_per_row / QK_TURBO3) * sizeof(block_turbo3_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo3_0_ref(
            src + row * n_per_row,
            (block_turbo3_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TURBO2_0: 2-bit PolarQuant with WHT rotation ---------- */

void quantize_row_turbo2_0_ref(const float * GGML_RESTRICT x, block_turbo2_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO2 == 0);

    extern int turbo3_cpu_wht_group_size;
    int group_size = turbo3_cpu_wht_group_size;
    if (group_size != 64 && group_size != 128) {
        group_size = (k % 128 == 0) ? 128 : 64;
    }
    if (k % group_size != 0) group_size = (group_size == 128) ? 64 : 128;
    assert(k % group_size == 0);

    const int n_groups = k / group_size;
    const int blocks_per_group = group_size / QK_TURBO2;

    for (int g = 0; g < n_groups; g++) {
        const float * grp_src = x + g * group_size;
        block_turbo2_0 * grp_dst = y + g * blocks_per_group;

        float norm_sq = 0.0f;
        float buf[128];
        for (int j = 0; j < group_size; j++) {
            buf[j] = grp_src[j];
            norm_sq += buf[j] * buf[j];
        }
        float grp_norm = sqrtf(norm_sq);
        float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

        for (int j = 0; j < group_size; j++) buf[j] *= inv_norm;

        turbo_cpu_fwht(buf, group_size);

        float recon_sq = 0.0f;
        for (int b = 0; b < blocks_per_group; b++) {
            block_turbo2_0 * blk = &grp_dst[b];
            const int off = b * QK_TURBO2;

            memset(blk->qs, 0, QK_TURBO2 / 4);

            for (int j = 0; j < QK_TURBO2; j++) {
                int idx = nearest_centroid_2bit(buf[off + j]);
                blk->qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
                recon_sq += CENTROIDS_2BIT[idx] * CENTROIDS_2BIT[idx];
            }
        }

        float recon_norm = sqrtf(recon_sq);
        float corrected = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        for (int b = 0; b < blocks_per_group; b++) {
            grp_dst[b].norm = GGML_FP32_TO_FP16(corrected);
        }
    }
}

void dequantize_row_turbo2_0(const block_turbo2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO2 == 0);
    const int nb = k / QK_TURBO2;
    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int j = 0; j < QK_TURBO2; j++) {
            uint8_t idx = (x[block].qs[j / 4] >> ((j % 4) * 2)) & 0x3;
            y[block * QK_TURBO2 + j] = CENTROIDS_2BIT[idx] * norm;
        }
    }
}

size_t quantize_turbo2_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO2 == 0);

    size_t row_size = (n_per_row / QK_TURBO2) * sizeof(block_turbo2_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo2_0_ref(
            src + row * n_per_row,
            (block_turbo2_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TURBO4_0: 4-bit PolarQuant with WHT rotation ---------- */
/*
 * Same pipeline as turbo3/turbo2 but with 16-centroid 4-bit quantization.
 * Struct layout: norm(fp16) + rnorm(fp16, unused/zero) + qs[64] (nibble-packed).
 * GPU kernel: cpy-turbo.cu k_cpy_f16/f32_turbo4 — same algorithm.
 * Dequantize: centroid[index] * norm (no inverse rotation needed as
 *   dequantized values are in WHT-rotated space, and attention operates
 *   in the same space after Q is also WHT-rotated).
 */

void quantize_row_turbo4_0_ref(const float * GGML_RESTRICT x, block_turbo4_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO4 == 0);

    extern int turbo3_cpu_wht_group_size;
    int group_size = turbo3_cpu_wht_group_size;
    if (group_size != 64 && group_size != 128) {
        group_size = (k % 128 == 0) ? 128 : 64;
    }
    if (k % group_size != 0) group_size = (group_size == 128) ? 64 : 128;
    assert(k % group_size == 0);

    const int n_groups = k / group_size;
    const int blocks_per_group = group_size / QK_TURBO4;

    for (int g = 0; g < n_groups; g++) {
        const float * grp_src = x + g * group_size;
        block_turbo4_0 * grp_dst = y + g * blocks_per_group;

        float norm_sq = 0.0f;
        float buf[128];
        for (int j = 0; j < group_size; j++) {
            buf[j] = grp_src[j];
            norm_sq += buf[j] * buf[j];
        }
        float grp_norm = sqrtf(norm_sq);
        float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

        for (int j = 0; j < group_size; j++) buf[j] *= inv_norm;

        turbo_cpu_fwht(buf, group_size);

        float recon_sq = 0.0f;
        for (int b = 0; b < blocks_per_group; b++) {
            block_turbo4_0 * blk = &grp_dst[b];
            const int off = b * QK_TURBO4;

            memset(blk->qs, 0, QK_TURBO4 / 2);

            for (int j = 0; j < QK_TURBO4; j++) {
                int idx = nearest_centroid_4bit(buf[off + j]);
                blk->qs[j / 2] |= (uint8_t)((idx & 0xF) << ((j % 2) * 4));
                recon_sq += CENTROIDS_4BIT[idx] * CENTROIDS_4BIT[idx];
            }
        }

        float recon_norm = sqrtf(recon_sq);
        float corrected = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        for (int b = 0; b < blocks_per_group; b++) {
            grp_dst[b].norm  = GGML_FP32_TO_FP16(corrected);
            grp_dst[b].rnorm = GGML_FP32_TO_FP16(0.0f);
        }
    }
}

void dequantize_row_turbo4_0(const block_turbo4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;
    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int j = 0; j < QK_TURBO4; j++) {
            uint8_t idx = (x[block].qs[j / 2] >> ((j % 2) * 4)) & 0xF;
            y[block * QK_TURBO4 + j] = CENTROIDS_4BIT[idx] * norm;
        }
    }
}

size_t quantize_turbo4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO4 == 0);

    size_t row_size = (n_per_row / QK_TURBO4) * sizeof(block_turbo4_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo4_0_ref(
            src + row * n_per_row,
            (block_turbo4_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TURBO3C_0: 3-bit integer-table quant with WHT (DP4A compatible) ---------- */

void quantize_row_turbo3c_0_ref(const float * GGML_RESTRICT x, block_turbo3c_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO3C == 0);

    extern int turbo3_cpu_wht_group_size;
    int group_size = turbo3_cpu_wht_group_size;
    if (group_size != 64 && group_size != 128) {
        group_size = (k % 128 == 0) ? 128 : 64;
    }
    if (k % group_size != 0) group_size = (group_size == 128) ? 64 : 128;
    assert(k % group_size == 0);

    const int n_groups = k / group_size;
    const int blocks_per_group = group_size / QK_TURBO3C;

    for (int g = 0; g < n_groups; g++) {
        const float * grp_src = x + g * group_size;
        block_turbo3c_0 * grp_dst = y + g * blocks_per_group;

        float norm_sq = 0.0f;
        float buf[128];
        for (int j = 0; j < group_size; j++) {
            buf[j] = grp_src[j];
            norm_sq += buf[j] * buf[j];
        }
        float grp_norm = sqrtf(norm_sq);
        float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

        for (int j = 0; j < group_size; j++) buf[j] *= inv_norm;

        turbo_cpu_fwht(buf, group_size);

        float recon_sq = 0.0f;
        for (int b = 0; b < blocks_per_group; b++) {
            block_turbo3c_0 * blk = &grp_dst[b];
            const int off = b * QK_TURBO3C;

            memset(blk->qs,    0, QK_TURBO3C / 4);
            memset(blk->signs, 0, QK_TURBO3C / 8);

            for (int j = 0; j < QK_TURBO3C; j++) {
                int idx = nearest_centroid_3cbit(buf[off + j]);
                blk->qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
                if (idx & 0x4) {
                    blk->signs[j / 8] |= (1 << (j % 8));
                }
                recon_sq += CENTROIDS_3CBIT[idx] * CENTROIDS_3CBIT[idx];
            }
        }

        float recon_norm = sqrtf(recon_sq);
        float corrected = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        for (int b = 0; b < blocks_per_group; b++) {
            grp_dst[b].norm = GGML_FP32_TO_FP16(corrected);
        }
    }
}

void dequantize_row_turbo3c_0(const block_turbo3c_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO3C == 0);
    const int nb = k / QK_TURBO3C;
    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int j = 0; j < QK_TURBO3C; j++) {
            uint8_t low2 = (x[block].qs[j / 4] >> ((j % 4) * 2)) & 0x3;
            uint8_t hi1  = (x[block].signs[j / 8] >> (j % 8)) & 0x1;
            uint8_t idx  = low2 | (hi1 << 2);
            y[block * QK_TURBO3C + j] = CENTROIDS_3CBIT[idx] * norm;
        }
    }
}

size_t quantize_turbo3c_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                           int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO3C == 0);

    size_t row_size = (n_per_row / QK_TURBO3C) * sizeof(block_turbo3c_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo3c_0_ref(
            src + row * n_per_row,
            (block_turbo3c_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

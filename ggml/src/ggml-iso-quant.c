/*
 * IsoQuant: KV cache compression via quaternion 4D block rotation + Lloyd-Max
 * Based on: ParaMind2025/isoquant
 *
 * Uses quaternion sandwich product T(v) = q_L * v for 4D block rotation.
 * 16 FMAs per quaternion multiply (4 groups of 4 elements = 32 groups for d=128).
 * Better decorrelation than PlanarQuant (2D) but cheaper than WHT (d log d).
 */

#define _USE_MATH_DEFINES

#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>

#define ISO_N_GROUPS 32  /* 128 / 4 */

static const float ISO_CENTROIDS_3BIT[8] = {
    -0.1906850000f, -0.1178320000f, -0.0657170000f, -0.0214600000f,
    0.0214600000f, 0.0657170000f, 0.1178320000f, 0.1906850000f,
};

/* Unit quaternions (one per 4D group, lazy init) */
static float iso_qw[ISO_N_GROUPS];
static float iso_qx[ISO_N_GROUPS];
static float iso_qy[ISO_N_GROUPS];
static float iso_qz[ISO_N_GROUPS];
static pthread_once_t iso_rotation_once = PTHREAD_ONCE_INIT;

static void iso_init_rotation_impl(void) {
    /* Must match planar-iso-constants.cuh PI_QW/QX/QY/QZ exactly */
    static const float QW[]={0.8350809813f,-0.1648498178f,0.1283752173f,0.2897698581f,-0.1820549369f,0.9549587369f,-0.8741137385f,0.8988990188f,-0.1312584430f,-0.3990598321f,-0.2694816887f,-0.1181898862f,0.1363395452f,0.2665117681f,-0.8263269663f,-0.1834189594f,0.3098247349f,0.2804697454f,-0.5655074716f,-0.1627507508f,0.8684155941f,0.2233296037f,-0.1291671842f,0.6606932878f,-0.5694432259f,-0.2782760859f,0.5113853812f,-0.5139024258f,0.7489815354f,-0.3037399948f,-0.4143463373f,-0.3524050117f};
    static const float QX[]={0.3547102809f,-0.5782636404f,-0.8299785256f,0.5694668293f,-0.8199930191f,0.1259543896f,-0.3090814352f,-0.2613596618f,-0.1660282463f,-0.5143862963f,0.5898610353f,-0.8277072310f,-0.6826571226f,-0.1740629375f,0.1416199356f,0.4648889899f,0.3485621810f,0.8982698917f,-0.3015249372f,0.4990116358f,0.2398942262f,-0.7447698116f,0.4783197045f,0.0735855624f,-0.2975912094f,-0.0700704753f,0.2975627482f,-0.2652103305f,-0.1539765000f,0.0849994123f,-0.1069803685f,-0.5753474832f};
    static const float QY[]={0.2416850179f,-0.4488199651f,0.3478420675f,0.5024775267f,0.1696543097f,0.1760476083f,0.0254505407f,0.2389279008f,-0.9429193735f,0.3925755024f,-0.2757458389f,-0.1485267133f,0.5530825853f,-0.8936085105f,0.2953715622f,-0.5285226703f,0.7939327955f,0.0139789311f,-0.2555710375f,0.4543992281f,-0.2698826790f,-0.4736968279f,0.4361720681f,-0.3461222053f,0.0792116225f,0.8827795386f,0.7416539788f,-0.3826399446f,-0.3534849286f,-0.8696597815f,-0.6908422709f,0.2082736641f};
    static const float QZ[]={0.3038694561f,0.4734756052f,-0.3878843784f,0.5831694603f,-0.5054479241f,-0.1731694490f,-0.3737666607f,0.2328704894f,0.2621760964f,0.6239953637f,-0.7082104683f,0.5308507681f,-0.4413037896f,-0.2802782655f,-0.4522367120f,-0.6698107123f,-0.3752456903f,-0.3359423280f,0.7181019187f,0.7106907368f,0.3100073636f,0.4016827941f,0.7350437641f,-0.6607965231f,0.7619289756f,0.3648703992f,-0.3040413559f,0.7213236690f,0.5280022621f,-0.3742936850f,-0.5760775208f,0.7015634775f};
    for(int i=0;i<ISO_N_GROUPS;i++){iso_qw[i]=QW[i];iso_qx[i]=QX[i];iso_qy[i]=QY[i];iso_qz[i]=QZ[i];}
}

static void iso_init_rotation(void) {
    pthread_once(&iso_rotation_once, iso_init_rotation_impl);
}

/* Hamilton product: q * v where v = (0, v1, v2, v3) treated as pure quaternion
 * Returns (rw, rx, ry, rz) */
static void quat_mul(float aw, float ax, float ay, float az,
                     float bw, float bx, float by, float bz,
                     float *rw, float *rx, float *ry, float *rz) {
    *rw = aw*bw - ax*bx - ay*by - az*bz;
    *rx = aw*bx + ax*bw + ay*bz - az*by;
    *ry = aw*by - ax*bz + ay*bw + az*bx;
    *rz = aw*bz + ax*by - ay*bx + az*bw;
}

/* Midpoints between adjacent 3-bit centroids for O(1) nearest-centroid lookup */
static const float ISO_MIDPOINTS_3BIT[7] = {
    -0.154259f, -0.091775f, -0.043589f, 0.000000f, 0.043589f, 0.091775f, 0.154259f
};

static int nearest_centroid_iso3(float val) {
    if      (val < ISO_MIDPOINTS_3BIT[0]) return 0;
    else if (val < ISO_MIDPOINTS_3BIT[1]) return 1;
    else if (val < ISO_MIDPOINTS_3BIT[2]) return 2;
    else if (val < ISO_MIDPOINTS_3BIT[3]) return 3;
    else if (val < ISO_MIDPOINTS_3BIT[4]) return 4;
    else if (val < ISO_MIDPOINTS_3BIT[5]) return 5;
    else if (val < ISO_MIDPOINTS_3BIT[6]) return 6;
    else                                  return 7;
}

void quantize_row_iso3_0_ref(const float * GGML_RESTRICT x, block_iso3_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_ISO3 == 0);
    iso_init_rotation();

    const int nb = k / QK_ISO3;

    for (int block = 0; block < nb; block++) {
        const float * src = x + block * QK_ISO3;
        block_iso3_0 * blk = &y[block];

        /* 1. L2 norm */
        float norm_sq = 0.0f;
        for (int j = 0; j < QK_ISO3; j++) norm_sq += src[j] * src[j];
        float grp_norm = sqrtf(norm_sq);
        float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

        /* 2. Normalize + rotate + quantize */
        memset(blk->qs, 0, QK_ISO3 / 4);
        memset(blk->signs, 0, QK_ISO3 / 8);

        float recon_sq = 0.0f;
        for (int g = 0; g < ISO_N_GROUPS; g++) {
            /* Load 4D block as quaternion (w=0, x=v0, y=v1, z=v2... wait,
             * we treat 4 elements as a quaternion: (v0, v1, v2, v3) */
            float v0 = src[g*4 + 0] * inv_norm;
            float v1 = src[g*4 + 1] * inv_norm;
            float v2 = src[g*4 + 2] * inv_norm;
            float v3 = src[g*4 + 3] * inv_norm;

            /* Forward rotation: rotated = q_L * v (left multiply) */
            float rw, rx, ry, rz;
            quat_mul(iso_qw[g], iso_qx[g], iso_qy[g], iso_qz[g],
                     v0, v1, v2, v3, &rw, &rx, &ry, &rz);

            /* Quantize all 4 components */
            float rotated[4] = {rw, rx, ry, rz};
            for (int c = 0; c < 4; c++) {
                int j = g * 4 + c;
                int idx = nearest_centroid_iso3(rotated[c]);
                blk->qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
                if (idx & 0x4) blk->signs[j / 8] |= (1 << (j % 8));
                recon_sq += ISO_CENTROIDS_3BIT[idx] * ISO_CENTROIDS_3BIT[idx];
            }
        }

        /* 3. Corrected norm */
        float recon_norm = sqrtf(recon_sq);
        float corrected = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        blk->norm = GGML_FP32_TO_FP16(corrected);
    }
}

void dequantize_row_iso3_0(const block_iso3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_ISO3 == 0);
    iso_init_rotation();

    const int nb = k / QK_ISO3;

    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);

        for (int g = 0; g < ISO_N_GROUPS; g++) {
            /* Unpack 4 indices */
            float qvals[4];
            for (int c = 0; c < 4; c++) {
                int j = g * 4 + c;
                uint8_t low = (x[block].qs[j / 4] >> ((j % 4) * 2)) & 0x3;
                uint8_t hi = (x[block].signs[j / 8] >> (j % 8)) & 0x1;
                uint8_t idx = low | (hi << 2);
                qvals[c] = ISO_CENTROIDS_3BIT[idx];
            }

            /* Inverse rotation: conj(q_L) * v
             * conj(q) = (w, -x, -y, -z) */
            float rw, rx, ry, rz;
            quat_mul(iso_qw[g], -iso_qx[g], -iso_qy[g], -iso_qz[g],
                     qvals[0], qvals[1], qvals[2], qvals[3],
                     &rw, &rx, &ry, &rz);

            y[block * QK_ISO3 + g*4 + 0] = rw * norm;
            y[block * QK_ISO3 + g*4 + 1] = rx * norm;
            y[block * QK_ISO3 + g*4 + 2] = ry * norm;
            y[block * QK_ISO3 + g*4 + 3] = rz * norm;
        }
    }
}

size_t quantize_iso3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                       int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    assert(n_per_row % QK_ISO3 == 0);

    size_t row_size = (n_per_row / QK_ISO3) * sizeof(block_iso3_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_iso3_0_ref(
            src + row * n_per_row,
            (block_iso3_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

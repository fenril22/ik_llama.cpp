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

/* Lloyd-Max optimal centroids for absmax-normalized blocks.
 * Distribution: N(0, 0.3) clipped to [-1,+1], matching absmax normalization. */
static const float ISO_CENTROIDS_3BIT[8] = {
    -0.633480296f, -0.397661053f, -0.224092943f, -0.072697365f,
     0.072697365f,  0.224092943f,  0.397661053f,  0.633480296f,
};

/* Unit quaternions (one per 4D group, lazy init) */
static float iso_qw[ISO_N_GROUPS];
static float iso_qx[ISO_N_GROUPS];
static float iso_qy[ISO_N_GROUPS];
static float iso_qz[ISO_N_GROUPS];
static pthread_once_t iso_rotation_once = PTHREAD_ONCE_INIT;

static void iso_init_rotation_impl(void) {
    /* Must match planar-iso-constants.cuh PI_QW/QX/QY/QZ exactly (L2-normalized) */
    static const float QW[]={0.8461847305f,-0.1858021224f,0.1298912615f,0.2896497761f,-0.1829912520f,0.9603561648f,-0.8741332284f,0.9045046333f,-0.1310863473f,-0.4057431458f,-0.2697307305f,-0.1180170686f,0.1373557118f,0.2694350316f,-0.8285526039f,-0.1854944856f,0.3116019500f,0.2806656152f,-0.5678682057f,-0.1638140456f,0.8769476586f,0.2244211569f,-0.1307450370f,0.6612280800f,-0.5695467902f,-0.2790055827f,0.5133009069f,-0.5136063182f,0.7533561361f,-0.3043666028f,-0.4159600566f,-0.3540403390f};
    static const float QX[]={0.3594267265f,-0.6517605730f,-0.8397801383f,0.5692308396f,-0.8242102725f,0.1266662840f,-0.3090883267f,-0.2629895240f,-0.1658105632f,-0.5230010571f,0.5904061561f,-0.8264969553f,-0.6877451062f,-0.1759721659f,0.1420013762f,0.4701495654f,0.3505616016f,0.8988972106f,-0.3027836654f,0.5022718143f,0.2422511542f,-0.7484099734f,0.4841626597f,0.0736451256f,-0.2976453321f,-0.0702541640f,0.2986773461f,-0.2650575178f,-0.1548758355f,0.0851747640f,-0.1073970158f,-0.5780173699f};
    static const float QY[]={0.2448986102f,-0.5058646907f,0.3519499005f,0.5022692977f,0.1705268479f,0.1770426296f,0.0254511082f,0.2404178765f,-0.9416830922f,0.3991502188f,-0.2760006698f,-0.1483095371f,0.5572048233f,-0.9034101534f,0.2961671190f,-0.5345033097f,0.7984869488f,0.0139886935f,-0.2566379293f,0.4573679416f,-0.2725342394f,-0.4760120843f,0.4415001652f,-0.3464023708f,0.0792260287f,0.8850937325f,0.7444320349f,-0.3824194697f,-0.3555495395f,-0.8714538678f,-0.6935328353f,0.2092401532f};
    static const float QZ[]={0.3079098909f,0.5336540467f,-0.3924650902f,0.5829277922f,-0.5080474609f,-0.1741482029f,-0.3737749945f,0.2343226905f,0.2618323518f,0.6344458186f,-0.7088649619f,0.5300745567f,-0.4445929173f,-0.2833525284f,-0.4534547710f,-0.6773901342f,-0.3773981727f,-0.3361769379f,0.7210996645f,0.7153338724f,0.3130531436f,0.4036460723f,0.7440227539f,-0.6613313989f,0.7620675472f,0.3658269017f,-0.3051802211f,0.7209080464f,0.5310861820f,-0.3750658435f,-0.5783211207f,0.7048190666f};
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

static const float ISO_MIDPOINTS_3BIT[7] = {
    -0.515570675f, -0.310876998f, -0.148395154f, 0.000000f, 0.148395154f, 0.310876998f, 0.515570675f
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

        /* 1. absmax norm */
        float grp_norm = 0.0f;
        for (int j = 0; j < QK_ISO3; j++) {
            float av = src[j] < 0 ? -src[j] : src[j];
            if (av > grp_norm) grp_norm = av;
        }
        float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

        /* 2. Normalize + rotate + quantize */
        memset(blk->qs, 0, QK_ISO3 / 4);
        memset(blk->signs, 0, QK_ISO3 / 8);

        for (int g = 0; g < ISO_N_GROUPS; g++) {
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
            }
        }

        /* 3. Store absmax as norm */
        blk->norm = GGML_FP32_TO_FP16(grp_norm);
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

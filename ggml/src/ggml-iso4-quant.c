/*
 * IsoQuant 4-bit: quaternion 4D rotation + 4-bit (16 centroids) nibble packed.
 * Same block layout as turbo4_0 but uses quaternion rotation instead of WHT.
 */

#define _USE_MATH_DEFINES

#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"
#include <math.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>

#define ISO4_N_GROUPS 32

/* Lloyd-Max optimal centroids for N(0, 1/sqrt(128)) */
static const float ISO4_CENTROIDS[16] = {
    -0.240803750f, -0.182222715f, -0.142468764f, -0.110596604f,
    -0.082955822f, -0.057812915f, -0.034158020f, -0.011301852f,
     0.011301852f,  0.034158020f,  0.057812915f,  0.082955822f,
     0.110596604f,  0.142468764f,  0.182222715f,  0.240803750f,
};

static float i4_qw[32], i4_qx[32], i4_qy[32], i4_qz[32];
static pthread_once_t i4_once = PTHREAD_ONCE_INIT;

static void iso4_init_impl(void) {
    /* Must match planar-iso-constants.cuh PI_QW/QX/QY/QZ exactly */
    static const float QW[]={0.8350809813f,-0.1648498178f,0.1283752173f,0.2897698581f,-0.1820549369f,0.9549587369f,-0.8741137385f,0.8988990188f,-0.1312584430f,-0.3990598321f,-0.2694816887f,-0.1181898862f,0.1363395452f,0.2665117681f,-0.8263269663f,-0.1834189594f,0.3098247349f,0.2804697454f,-0.5655074716f,-0.1627507508f,0.8684155941f,0.2233296037f,-0.1291671842f,0.6606932878f,-0.5694432259f,-0.2782760859f,0.5113853812f,-0.5139024258f,0.7489815354f,-0.3037399948f,-0.4143463373f,-0.3524050117f};
    static const float QX[]={0.3547102809f,-0.5782636404f,-0.8299785256f,0.5694668293f,-0.8199930191f,0.1259543896f,-0.3090814352f,-0.2613596618f,-0.1660282463f,-0.5143862963f,0.5898610353f,-0.8277072310f,-0.6826571226f,-0.1740629375f,0.1416199356f,0.4648889899f,0.3485621810f,0.8982698917f,-0.3015249372f,0.4990116358f,0.2398942262f,-0.7447698116f,0.4783197045f,0.0735855624f,-0.2975912094f,-0.0700704753f,0.2975627482f,-0.2652103305f,-0.1539765000f,0.0849994123f,-0.1069803685f,-0.5753474832f};
    static const float QY[]={0.2416850179f,-0.4488199651f,0.3478420675f,0.5024775267f,0.1696543097f,0.1760476083f,0.0254505407f,0.2389279008f,-0.9429193735f,0.3925755024f,-0.2757458389f,-0.1485267133f,0.5530825853f,-0.8936085105f,0.2953715622f,-0.5285226703f,0.7939327955f,0.0139789311f,-0.2555710375f,0.4543992281f,-0.2698826790f,-0.4736968279f,0.4361720681f,-0.3461222053f,0.0792116225f,0.8827795386f,0.7416539788f,-0.3826399446f,-0.3534849286f,-0.8696597815f,-0.6908422709f,0.2082736641f};
    static const float QZ[]={0.3038694561f,0.4734756052f,-0.3878843784f,0.5831694603f,-0.5054479241f,-0.1731694490f,-0.3737666607f,0.2328704894f,0.2621760964f,0.6239953637f,-0.7082104683f,0.5308507681f,-0.4413037896f,-0.2802782655f,-0.4522367120f,-0.6698107123f,-0.3752456903f,-0.3359423280f,0.7181019187f,0.7106907368f,0.3100073636f,0.4016827941f,0.7350437641f,-0.6607965231f,0.7619289756f,0.3648703992f,-0.3040413559f,0.7213236690f,0.5280022621f,-0.3742936850f,-0.5760775208f,0.7015634775f};
    for(int i=0;i<32;i++){i4_qw[i]=QW[i];i4_qx[i]=QX[i];i4_qy[i]=QY[i];i4_qz[i]=QZ[i];}
}

static void iso4_init(void) {
    pthread_once(&i4_once, iso4_init_impl);
}

static const float ISO4_MIDPOINTS[15] = {
    -0.211513233f, -0.162345739f, -0.126532684f, -0.096776213f, -0.070384368f, -0.045985467f, -0.022729936f,
     0.000000000f,
     0.022729936f,  0.045985467f,  0.070384368f,  0.096776213f,  0.126532684f,  0.162345739f,  0.211513233f
};

static int nearest_16(float val) {
    if      (val < ISO4_MIDPOINTS[0])  return 0;
    else if (val < ISO4_MIDPOINTS[1])  return 1;
    else if (val < ISO4_MIDPOINTS[2])  return 2;
    else if (val < ISO4_MIDPOINTS[3])  return 3;
    else if (val < ISO4_MIDPOINTS[4])  return 4;
    else if (val < ISO4_MIDPOINTS[5])  return 5;
    else if (val < ISO4_MIDPOINTS[6])  return 6;
    else if (val < ISO4_MIDPOINTS[7])  return 7;
    else if (val < ISO4_MIDPOINTS[8])  return 8;
    else if (val < ISO4_MIDPOINTS[9])  return 9;
    else if (val < ISO4_MIDPOINTS[10]) return 10;
    else if (val < ISO4_MIDPOINTS[11]) return 11;
    else if (val < ISO4_MIDPOINTS[12]) return 12;
    else if (val < ISO4_MIDPOINTS[13]) return 13;
    else if (val < ISO4_MIDPOINTS[14]) return 14;
    else                               return 15;
}

void quantize_row_iso4_0_ref(const float * GGML_RESTRICT x, block_iso4_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % 128 == 0);
    iso4_init();
    const int nb = k / 128;

    for (int b = 0; b < nb; b++) {
        const float * src = x + b * 128;
        block_iso4_0 * blk = &y[b];

        float norm_sq = 0;
        for (int j = 0; j < 128; j++) norm_sq += src[j] * src[j];
        float grp_norm = sqrtf(norm_sq);
        float inv = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

        memset(blk->qs, 0, 64);
        float recon_sq = 0;

        for (int g = 0; g < 32; g++) {
            float v0 = src[g*4]*inv, v1 = src[g*4+1]*inv, v2 = src[g*4+2]*inv, v3 = src[g*4+3]*inv;
            float qw=i4_qw[g], qx=i4_qx[g], qy=i4_qy[g], qz=i4_qz[g];
            /* q_L * v */
            float rw = qw*v0 - qx*v1 - qy*v2 - qz*v3;
            float rx = qw*v1 + qx*v0 + qy*v3 - qz*v2;
            float ry = qw*v2 - qx*v3 + qy*v0 + qz*v1;
            float rz = qw*v3 + qx*v2 - qy*v1 + qz*v0;

            float rot[4] = {rw, rx, ry, rz};
            for (int c = 0; c < 4; c++) {
                int j = g*4 + c;
                int idx = nearest_16(rot[c]);
                blk->qs[j/2] |= (idx & 0xF) << ((j%2)*4);
                recon_sq += ISO4_CENTROIDS[idx] * ISO4_CENTROIDS[idx];
            }
        }

        float rn = sqrtf(recon_sq);
        blk->norm = GGML_FP32_TO_FP16((rn > 1e-10f) ? grp_norm / rn : grp_norm);
        blk->rnorm = GGML_FP32_TO_FP16(0.0f);
    }
}

void dequantize_row_iso4_0(const block_iso4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % 128 == 0);
    iso4_init();
    const int nb = k / 128;

    for (int b = 0; b < nb; b++) {
        float norm = GGML_FP16_TO_FP32(x[b].norm);
        for (int g = 0; g < 32; g++) {
            float qvals[4];
            for (int c = 0; c < 4; c++) {
                int j = g*4 + c;
                uint8_t idx = (x[b].qs[j/2] >> ((j%2)*4)) & 0xF;
                qvals[c] = ISO4_CENTROIDS[idx];
            }
            /* conj(q_L) * v */
            float qw=i4_qw[g], qx=-i4_qx[g], qy=-i4_qy[g], qz=-i4_qz[g];
            float rw = qw*qvals[0] - qx*qvals[1] - qy*qvals[2] - qz*qvals[3];
            float rx = qw*qvals[1] + qx*qvals[0] + qy*qvals[3] - qz*qvals[2];
            float ry = qw*qvals[2] - qx*qvals[3] + qy*qvals[0] + qz*qvals[1];
            float rz = qw*qvals[3] + qx*qvals[2] - qy*qvals[1] + qz*qvals[0];

            y[b*128 + g*4]   = rw * norm;
            y[b*128 + g*4+1] = rx * norm;
            y[b*128 + g*4+2] = ry * norm;
            y[b*128 + g*4+3] = rz * norm;
        }
    }
}

size_t quantize_iso4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                       int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    assert(n_per_row % 128 == 0);
    size_t row_size = (n_per_row / 128) * sizeof(block_iso4_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_iso4_0_ref(
            src + row * n_per_row,
            (block_iso4_0 *)((char *)dst + row * row_size),
            n_per_row);
    }
    return nrows * row_size;
}

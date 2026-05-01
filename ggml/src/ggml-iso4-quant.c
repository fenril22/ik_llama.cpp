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

/* Lloyd-Max optimal centroids for absmax-normalized blocks.
 * Distribution: N(0, 0.3) clipped to [-1,+1], 16 levels. */
static const float ISO4_CENTROIDS[16] = {
    -0.678649914f, -0.514937045f, -0.403020127f, -0.313029964f,
    -0.234871876f, -0.163717400f, -0.096741518f, -0.032010552f,
     0.032010552f,  0.096741518f,  0.163717400f,  0.234871876f,
     0.313029964f,  0.403020127f,  0.514937045f,  0.678649914f,
};

static float i4_qw[32], i4_qx[32], i4_qy[32], i4_qz[32];
static pthread_once_t i4_once = PTHREAD_ONCE_INIT;

static void iso4_init_impl(void) {
    /* Must match planar-iso-constants.cuh PI_QW/QX/QY/QZ exactly (L2-normalized) */
    static const float QW[]={0.8461847305f,-0.1858021224f,0.1298912615f,0.2896497761f,-0.1829912520f,0.9603561648f,-0.8741332284f,0.9045046333f,-0.1310863473f,-0.4057431458f,-0.2697307305f,-0.1180170686f,0.1373557118f,0.2694350316f,-0.8285526039f,-0.1854944856f,0.3116019500f,0.2806656152f,-0.5678682057f,-0.1638140456f,0.8769476586f,0.2244211569f,-0.1307450370f,0.6612280800f,-0.5695467902f,-0.2790055827f,0.5133009069f,-0.5136063182f,0.7533561361f,-0.3043666028f,-0.4159600566f,-0.3540403390f};
    static const float QX[]={0.3594267265f,-0.6517605730f,-0.8397801383f,0.5692308396f,-0.8242102725f,0.1266662840f,-0.3090883267f,-0.2629895240f,-0.1658105632f,-0.5230010571f,0.5904061561f,-0.8264969553f,-0.6877451062f,-0.1759721659f,0.1420013762f,0.4701495654f,0.3505616016f,0.8988972106f,-0.3027836654f,0.5022718143f,0.2422511542f,-0.7484099734f,0.4841626597f,0.0736451256f,-0.2976453321f,-0.0702541640f,0.2986773461f,-0.2650575178f,-0.1548758355f,0.0851747640f,-0.1073970158f,-0.5780173699f};
    static const float QY[]={0.2448986102f,-0.5058646907f,0.3519499005f,0.5022692977f,0.1705268479f,0.1770426296f,0.0254511082f,0.2404178765f,-0.9416830922f,0.3991502188f,-0.2760006698f,-0.1483095371f,0.5572048233f,-0.9034101534f,0.2961671190f,-0.5345033097f,0.7984869488f,0.0139886935f,-0.2566379293f,0.4573679416f,-0.2725342394f,-0.4760120843f,0.4415001652f,-0.3464023708f,0.0792260287f,0.8850937325f,0.7444320349f,-0.3824194697f,-0.3555495395f,-0.8714538678f,-0.6935328353f,0.2092401532f};
    static const float QZ[]={0.3079098909f,0.5336540467f,-0.3924650902f,0.5829277922f,-0.5080474609f,-0.1741482029f,-0.3737749945f,0.2343226905f,0.2618323518f,0.6344458186f,-0.7088649619f,0.5300745567f,-0.4445929173f,-0.2833525284f,-0.4534547710f,-0.6773901342f,-0.3773981727f,-0.3361769379f,0.7210996645f,0.7153338724f,0.3130531436f,0.4036460723f,0.7440227539f,-0.6613313989f,0.7620675472f,0.3658269017f,-0.3051802211f,0.7209080464f,0.5310861820f,-0.3750658435f,-0.5783211207f,0.7048190666f};
    for(int i=0;i<32;i++){i4_qw[i]=QW[i];i4_qx[i]=QX[i];i4_qy[i]=QY[i];i4_qz[i]=QZ[i];}
}

static void iso4_init(void) {
    pthread_once(&i4_once, iso4_init_impl);
}

static const float ISO4_MIDPOINTS[15] = {
    -0.596793479f, -0.458978586f, -0.358025046f, -0.273950920f, -0.199294638f, -0.130229459f, -0.064376035f,
     0.000000000f,
     0.064376035f,  0.130229459f,  0.199294638f,  0.273950920f,  0.358025046f,  0.458978586f,  0.596793479f
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

        float grp_norm = 0.0f;
        for (int j = 0; j < 128; j++) {
            float av = src[j] < 0 ? -src[j] : src[j];
            if (av > grp_norm) grp_norm = av;
        }
        float inv = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

        memset(blk->qs, 0, 64);

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
            }
        }

        blk->norm = GGML_FP32_TO_FP16(grp_norm);
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

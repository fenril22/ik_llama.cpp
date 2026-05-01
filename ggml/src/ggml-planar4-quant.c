/*
 * PlanarQuant 4-bit: 2D Givens rotation + 4-bit (16 centroids) nibble packed.
 * Same block layout as turbo4_0 but uses Givens rotation instead of WHT.
 */
#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"
#include <math.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>


/* Lloyd-Max optimal centroids for N(0, 1/sqrt(128)) */
static const float PLANAR4_CENTROIDS[16] = {
    -0.240803750f, -0.182222715f, -0.142468764f, -0.110596604f,
    -0.082955822f, -0.057812915f, -0.034158020f, -0.011301852f,
     0.011301852f,  0.034158020f,  0.057812915f,  0.082955822f,
     0.110596604f,  0.142468764f,  0.182222715f,  0.240803750f,
};

static float p4_cos[64], p4_sin[64];
static pthread_once_t p4_once = PTHREAD_ONCE_INIT;

static void planar4_init_impl(void) {
    /* Must match planar-iso-constants.cuh PI_COS/PI_SIN exactly */
    static const float COS[]={-0.9095053397f,0.1535578452f,-0.8537489227f,-0.6827218011f,-0.4249387949f,0.9864510046f,0.9906673944f,0.5752363372f,-0.9866459035f,0.9878848090f,-0.6215683804f,-0.9835597698f,0.8777263755f,-0.4624640047f,0.2843135922f,-0.7739960698f,0.2385234222f,0.9121914932f,-0.8815003943f,-0.2639699512f,-0.5517087300f,-0.9035294557f,-0.8520543188f,-0.5600635985f,-0.7667286376f,-0.9877949369f,-0.9781949787f,-0.9953372831f,-0.8622053901f,-0.7382118186f,0.9136037642f,-0.2558504503f,-0.8541000475f,-0.6159335408f,0.9861256679f,-0.6758560284f,0.4249571682f,-0.6219544719f,0.9130573430f,-0.5948161096f,0.5759782996f,0.9729901203f,0.6535998325f,0.9222195491f,-0.7668084044f,0.5116178563f,-0.7848786574f,0.9902111051f,0.1997167840f,0.7173003220f,-0.9999998006f,-0.9557868691f,0.5594852693f,-0.9980111824f,0.9782398557f,-0.9150004329f,-0.4084754305f,0.0071549185f,0.9558482753f,-0.0971921648f,-0.9469334002f,0.9999492419f,0.6100589016f,0.0350818915f};
    static const float SIN[]={-0.4156922383f,0.9881396603f,0.5206849114f,-0.7306784124f,-0.9052220836f,0.1640561354f,0.1363015542f,0.8179872593f,0.1628798979f,0.1551889303f,0.7833599099f,-0.1805828875f,-0.4791621957f,0.8866380571f,-0.9587313395f,0.6331904010f,-0.9711367448f,0.4097641756f,0.4721832852f,-0.9645309040f,0.8340368561f,0.4285259884f,0.5234533769f,0.8284496156f,0.6419713361f,-0.1557599517f,-0.2076886701f,0.0964556523f,0.5065588468f,-0.6745689815f,-0.4066056591f,-0.9667163736f,0.5201087471f,-0.7877981171f,0.1660005034f,-0.7370336688f,0.9052134584f,0.7830534049f,-0.4078312009f,-0.8038618014f,0.8174649829f,-0.2308467584f,-0.7568403127f,-0.3866666566f,0.6418760557f,-0.8592131104f,0.6196494922f,0.1395778183f,0.9798536657f,0.6967641265f,-0.0006314605f,0.2940603015f,0.8288402943f,-0.0630371303f,0.2074771907f,0.4034528570f,0.9127693152f,-0.9999744032f,0.2938606379f,0.9952656344f,0.3214298299f,0.0100754012f,-0.7923560668f,-0.9993844410f};
    for(int i=0;i<64;i++){p4_cos[i]=COS[i];p4_sin[i]=SIN[i];}
}

static void planar4_init(void) {
    pthread_once(&p4_once, planar4_init_impl);
}

static const float PLANAR4_MIDPOINTS[15] = {
    -0.211513233f, -0.162345739f, -0.126532684f, -0.096776213f, -0.070384368f, -0.045985467f, -0.022729936f,
     0.000000000f,
     0.022729936f,  0.045985467f,  0.070384368f,  0.096776213f,  0.126532684f,  0.162345739f,  0.211513233f
};

static int nearest_4bit(float val) {
    if      (val < PLANAR4_MIDPOINTS[0])  return 0;
    else if (val < PLANAR4_MIDPOINTS[1])  return 1;
    else if (val < PLANAR4_MIDPOINTS[2])  return 2;
    else if (val < PLANAR4_MIDPOINTS[3])  return 3;
    else if (val < PLANAR4_MIDPOINTS[4])  return 4;
    else if (val < PLANAR4_MIDPOINTS[5])  return 5;
    else if (val < PLANAR4_MIDPOINTS[6])  return 6;
    else if (val < PLANAR4_MIDPOINTS[7])  return 7;
    else if (val < PLANAR4_MIDPOINTS[8])  return 8;
    else if (val < PLANAR4_MIDPOINTS[9])  return 9;
    else if (val < PLANAR4_MIDPOINTS[10]) return 10;
    else if (val < PLANAR4_MIDPOINTS[11]) return 11;
    else if (val < PLANAR4_MIDPOINTS[12]) return 12;
    else if (val < PLANAR4_MIDPOINTS[13]) return 13;
    else if (val < PLANAR4_MIDPOINTS[14]) return 14;
    else                                  return 15;
}

void quantize_row_planar4_0_ref(const float * GGML_RESTRICT x, block_planar4_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % 128 == 0);
    planar4_init();
    const int nb = k / 128;

    for (int b = 0; b < nb; b++) {
        const float * src = x + b * 128;
        block_planar4_0 * blk = &y[b];

        float norm_sq = 0.0f;
        for (int j = 0; j < 128; j++) norm_sq += src[j] * src[j];
        float grp_norm = sqrtf(norm_sq);
        float inv = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

        memset(blk->qs, 0, 64);

        float recon_sq = 0.0f;
        for (int p = 0; p < 64; p++) {
            float v0 = src[p*2] * inv;
            float v1 = src[p*2+1] * inv;
            float r0 = p4_cos[p]*v0 - p4_sin[p]*v1;
            float r1 = p4_sin[p]*v0 + p4_cos[p]*v1;

            int i0 = nearest_4bit(r0);
            int i1 = nearest_4bit(r1);

            int j0 = p*2, j1 = p*2+1;
            blk->qs[j0/2] |= (i0 & 0xF) << ((j0%2)*4);
            blk->qs[j1/2] |= (i1 & 0xF) << ((j1%2)*4);

            recon_sq += PLANAR4_CENTROIDS[i0]*PLANAR4_CENTROIDS[i0];
            recon_sq += PLANAR4_CENTROIDS[i1]*PLANAR4_CENTROIDS[i1];
        }

        float rn = sqrtf(recon_sq);
        float corrected = (rn > 1e-10f) ? grp_norm / rn : grp_norm;
        blk->norm = GGML_FP32_TO_FP16(corrected);
        blk->rnorm = GGML_FP32_TO_FP16(0.0f);
    }
}

void dequantize_row_planar4_0(const block_planar4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % 128 == 0);
    planar4_init();
    const int nb = k / 128;

    for (int b = 0; b < nb; b++) {
        float norm = GGML_FP16_TO_FP32(x[b].norm);
        for (int p = 0; p < 64; p++) {
            int j0 = p*2, j1 = p*2+1;
            uint8_t i0 = (x[b].qs[j0/2] >> ((j0%2)*4)) & 0xF;
            uint8_t i1 = (x[b].qs[j1/2] >> ((j1%2)*4)) & 0xF;
            float q0 = PLANAR4_CENTROIDS[i0];
            float q1 = PLANAR4_CENTROIDS[i1];
            float f0 =  p4_cos[p]*q0 + p4_sin[p]*q1;
            float f1 = -p4_sin[p]*q0 + p4_cos[p]*q1;
            y[b*128 + j0] = f0 * norm;
            y[b*128 + j1] = f1 * norm;
        }
    }
}

size_t quantize_planar4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                          int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    assert(n_per_row % 128 == 0);
    size_t row_size = (n_per_row / 128) * sizeof(block_planar4_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_planar4_0_ref(
            src + row * n_per_row,
            (block_planar4_0 *)((char *)dst + row * row_size),
            n_per_row);
    }
    return nrows * row_size;
}

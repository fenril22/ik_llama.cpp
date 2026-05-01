/*
 * PlanarQuant: KV cache compression via 2D Givens rotation + Lloyd-Max
 * Based on: ParaMind2025/isoquant (planar2_fused_kernel.cu)
 *
 * Instead of TurboQuant's dense d×d WHT rotation, uses independent
 * 2D Givens rotations per pair: only 4 FMAs per pair vs O(d log d) for WHT.
 * Same block layout as turbo3 (2-bit indices + 1-bit signs + norm).
 */

#define _USE_MATH_DEFINES

#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>

#define PLANAR_D 128

/* Same centroids as turbo3 (Lloyd-Max for N(0, 1/128)) */
static const float PLANAR_CENTROIDS_3BIT[8] = {
    -0.1906850000f, -0.1178320000f, -0.0657170000f, -0.0214600000f,
    0.0214600000f, 0.0657170000f, 0.1178320000f, 0.1906850000f,
};

/* Rotation parameters: cos/sin per pair (lazy init) */
static float planar_cos[PLANAR_D / 2];
static float planar_sin[PLANAR_D / 2];
static pthread_once_t planar_rotation_once = PTHREAD_ONCE_INIT;

static void planar_init_rotation_impl(void) {
    /* Must match planar-iso-constants.cuh PI_COS/PI_SIN exactly */
    static const float COS[]={-0.9095053397f,0.1535578452f,-0.8537489227f,-0.6827218011f,-0.4249387949f,0.9864510046f,0.9906673944f,0.5752363372f,-0.9866459035f,0.9878848090f,-0.6215683804f,-0.9835597698f,0.8777263755f,-0.4624640047f,0.2843135922f,-0.7739960698f,0.2385234222f,0.9121914932f,-0.8815003943f,-0.2639699512f,-0.5517087300f,-0.9035294557f,-0.8520543188f,-0.5600635985f,-0.7667286376f,-0.9877949369f,-0.9781949787f,-0.9953372831f,-0.8622053901f,-0.7382118186f,0.9136037642f,-0.2558504503f,-0.8541000475f,-0.6159335408f,0.9861256679f,-0.6758560284f,0.4249571682f,-0.6219544719f,0.9130573430f,-0.5948161096f,0.5759782996f,0.9729901203f,0.6535998325f,0.9222195491f,-0.7668084044f,0.5116178563f,-0.7848786574f,0.9902111051f,0.1997167840f,0.7173003220f,-0.9999998006f,-0.9557868691f,0.5594852693f,-0.9980111824f,0.9782398557f,-0.9150004329f,-0.4084754305f,0.0071549185f,0.9558482753f,-0.0971921648f,-0.9469334002f,0.9999492419f,0.6100589016f,0.0350818915f};
    static const float SIN[]={-0.4156922383f,0.9881396603f,0.5206849114f,-0.7306784124f,-0.9052220836f,0.1640561354f,0.1363015542f,0.8179872593f,0.1628798979f,0.1551889303f,0.7833599099f,-0.1805828875f,-0.4791621957f,0.8866380571f,-0.9587313395f,0.6331904010f,-0.9711367448f,0.4097641756f,0.4721832852f,-0.9645309040f,0.8340368561f,0.4285259884f,0.5234533769f,0.8284496156f,0.6419713361f,-0.1557599517f,-0.2076886701f,0.0964556523f,0.5065588468f,-0.6745689815f,-0.4066056591f,-0.9667163736f,0.5201087471f,-0.7877981171f,0.1660005034f,-0.7370336688f,0.9052134584f,0.7830534049f,-0.4078312009f,-0.8038618014f,0.8174649829f,-0.2308467584f,-0.7568403127f,-0.3866666566f,0.6418760557f,-0.8592131104f,0.6196494922f,0.1395778183f,0.9798536657f,0.6967641265f,-0.0006314605f,0.2940603015f,0.8288402943f,-0.0630371303f,0.2074771907f,0.4034528570f,0.9127693152f,-0.9999744032f,0.2938606379f,0.9952656344f,0.3214298299f,0.0100754012f,-0.7923560668f,-0.9993844410f};
    for(int i=0;i<PLANAR_D/2;i++){planar_cos[i]=COS[i];planar_sin[i]=SIN[i];}
}

static void planar_init_rotation(void) {
    pthread_once(&planar_rotation_once, planar_init_rotation_impl);
}

static int nearest_centroid_planar3(float val) {
    int best = 0;
    float best_d = fabsf(val - PLANAR_CENTROIDS_3BIT[0]);
    for (int i = 1; i < 8; i++) {
        float d = fabsf(val - PLANAR_CENTROIDS_3BIT[i]);
        if (d < best_d) { best_d = d; best = i; }
    }
    return best;
}

void quantize_row_planar3_0_ref(const float * GGML_RESTRICT x, block_planar3_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_PLANAR3 == 0);
    planar_init_rotation();

    const int nb = k / QK_PLANAR3;
    const int n_pairs = QK_PLANAR3 / 2;

    for (int block = 0; block < nb; block++) {
        const float * src = x + block * QK_PLANAR3;
        block_planar3_0 * blk = &y[block];

        /* 1. L2 norm */
        float norm_sq = 0.0f;
        for (int j = 0; j < QK_PLANAR3; j++) norm_sq += src[j] * src[j];
        float grp_norm = sqrtf(norm_sq);
        float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

        /* 2. Normalize + rotate + quantize */
        memset(blk->qs, 0, QK_PLANAR3 / 4);
        memset(blk->signs, 0, QK_PLANAR3 / 8);

        float recon_sq = 0.0f;
        for (int p = 0; p < n_pairs; p++) {
            float v0 = src[p * 2] * inv_norm;
            float v1 = src[p * 2 + 1] * inv_norm;

            /* Forward Givens rotation */
            float c = planar_cos[p];
            float s = planar_sin[p];
            float r0 = c * v0 - s * v1;
            float r1 = s * v0 + c * v1;

            /* Quantize both */
            int idx0 = nearest_centroid_planar3(r0);
            int idx1 = nearest_centroid_planar3(r1);

            int j0 = p * 2;
            int j1 = p * 2 + 1;

            /* Pack 2-bit lower + 1-bit sign (same as turbo3) */
            blk->qs[j0 / 4] |= (idx0 & 0x3) << ((j0 % 4) * 2);
            if (idx0 & 0x4) blk->signs[j0 / 8] |= (1 << (j0 % 8));

            blk->qs[j1 / 4] |= (idx1 & 0x3) << ((j1 % 4) * 2);
            if (idx1 & 0x4) blk->signs[j1 / 8] |= (1 << (j1 % 8));

            recon_sq += PLANAR_CENTROIDS_3BIT[idx0] * PLANAR_CENTROIDS_3BIT[idx0];
            recon_sq += PLANAR_CENTROIDS_3BIT[idx1] * PLANAR_CENTROIDS_3BIT[idx1];
        }

        /* 3. Corrected norm */
        float recon_norm = sqrtf(recon_sq);
        float corrected = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        blk->norm = GGML_FP32_TO_FP16(corrected);
    }
}

void dequantize_row_planar3_0(const block_planar3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_PLANAR3 == 0);
    planar_init_rotation();

    const int nb = k / QK_PLANAR3;
    const int n_pairs = QK_PLANAR3 / 2;

    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);

        for (int p = 0; p < n_pairs; p++) {
            int j0 = p * 2;
            int j1 = p * 2 + 1;

            /* Unpack indices */
            uint8_t low0 = (x[block].qs[j0 / 4] >> ((j0 % 4) * 2)) & 0x3;
            uint8_t hi0 = (x[block].signs[j0 / 8] >> (j0 % 8)) & 0x1;
            uint8_t idx0 = low0 | (hi0 << 2);

            uint8_t low1 = (x[block].qs[j1 / 4] >> ((j1 % 4) * 2)) & 0x3;
            uint8_t hi1 = (x[block].signs[j1 / 8] >> (j1 % 8)) & 0x1;
            uint8_t idx1 = low1 | (hi1 << 2);

            float q0 = PLANAR_CENTROIDS_3BIT[idx0];
            float q1 = PLANAR_CENTROIDS_3BIT[idx1];

            /* Inverse Givens rotation */
            float c = planar_cos[p];
            float s = planar_sin[p];
            float f0 = c * q0 + s * q1;
            float f1 = -s * q0 + c * q1;

            y[block * QK_PLANAR3 + j0] = f0 * norm;
            y[block * QK_PLANAR3 + j1] = f1 * norm;
        }
    }
}

size_t quantize_planar3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                          int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    assert(n_per_row % QK_PLANAR3 == 0);

    size_t row_size = (n_per_row / QK_PLANAR3) * sizeof(block_planar3_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_planar3_0_ref(
            src + row * n_per_row,
            (block_planar3_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

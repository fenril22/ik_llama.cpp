// ── InnerQ state definitions ──────────────────────────────────────────────────
// Device-side variables are declared `extern __device__` in turbo-quant.cuh and
// must be defined exactly once.  Host-side variables/functions are also defined
// here so that they are not multiply-defined when turbo-quant.cuh is included
// from other translation units (cpy-turbo.cu, set-rows.cu, fattn instances).

#include "turbo-innerq.cuh"  // INNERQ_MAX_CHANNELS

// Device state (one definition across the entire CUDA binary)
__device__ float d_innerq_scale[INNERQ_MAX_CHANNELS];
__device__ float d_innerq_scale_inv[INNERQ_MAX_CHANNELS];
__device__ float d_innerq_sq_accum[INNERQ_MAX_CHANNELS];
__device__ int   d_innerq_count;
__device__ int   d_innerq_active;
__device__ int   d_innerq_calibrating;

// Host state
int   innerq_enabled       = 0;
int   innerq_target_tokens = 0;
float innerq_strength      = 0.5f;
bool  innerq_initialized   = false;

// ─────────────────────────────────────────────────────────────────────────────
// Now pull in the rest of the turbo headers (they will see the extern decls).
#include "turbo-quant.cuh"
#include "turbo-wht.cuh"

// ── InnerQ host function bodies ───────────────────────────────────────────────
// These are declared in turbo-quant.cuh (non-static) and defined here.

#include <cstdlib>  // getenv / atoi / atof
#include <cmath>    // sqrtf / powf

void turbo_innerq_init(void) {
    if (innerq_initialized) return;
    innerq_initialized = true;

    const char * env = getenv("TURBO_INNERQ");
    if (!env || atoi(env) <= 0) {
        innerq_enabled = 0;
        return;
    }
    innerq_target_tokens = atoi(env);
    innerq_enabled = 1;  // calibrating

    const char * env_str = getenv("TURBO_INNERQ_STRENGTH");
    if (env_str) innerq_strength = atof(env_str);
    if (innerq_strength <= 0.0f || innerq_strength > 1.0f) innerq_strength = 0.5f;

    float zeros[INNERQ_MAX_CHANNELS] = {};
    int zero = 0, one = 1;
    cudaMemcpyToSymbol(d_innerq_sq_accum, zeros, sizeof(zeros));
    cudaMemcpyToSymbol(d_innerq_count, &zero, sizeof(int));
    cudaMemcpyToSymbol(d_innerq_active, &zero, sizeof(int));
    cudaMemcpyToSymbol(d_innerq_calibrating, &one, sizeof(int));

    fprintf(stderr, "[turbo-quant] %s: InnerQ calibration started (target=%d tokens, strength=%.2f)\n",
            __func__, innerq_target_tokens, innerq_strength);
}

void turbo_innerq_finalize(int group_size) {
    float sq_accum[INNERQ_MAX_CHANNELS];
    int count = 0;
    cudaMemcpyFromSymbol(sq_accum, d_innerq_sq_accum, group_size * sizeof(float));
    cudaMemcpyFromSymbol(&count, d_innerq_count, sizeof(int));

    if (count <= 0) {
        fprintf(stderr, "[turbo-quant WARN] %s: InnerQ calibration got 0 tokens, disabling\n", __func__);
        innerq_enabled = 0;
        int zero = 0;
        cudaMemcpyToSymbol(d_innerq_calibrating, &zero, sizeof(int));
        return;
    }

    float rms[INNERQ_MAX_CHANNELS];
    float mean_rms = 0.0f;
    float max_ratio = 0.0f, min_ratio = 1e30f;
    for (int i = 0; i < group_size; i++) {
        rms[i] = sqrtf(sq_accum[i] / (float)count);
        mean_rms += rms[i];
    }
    mean_rms /= (float)group_size;

    float scale[INNERQ_MAX_CHANNELS];
    float scale_inv[INNERQ_MAX_CHANNELS];
    for (int i = 0; i < group_size; i++) {
        float ratio = (rms[i] > 1e-10f) ? (mean_rms / rms[i]) : 1.0f;
        float s = powf(ratio, innerq_strength);
        if (s < 0.5f) s = 0.5f;
        if (s > 2.0f) s = 2.0f;
        scale[i] = s;
        scale_inv[i] = 1.0f / s;
        if (ratio > max_ratio) max_ratio = ratio;
        if (ratio < min_ratio) min_ratio = ratio;
    }

    if (max_ratio < 1.2f && min_ratio > (1.0f / 1.2f)) {
        fprintf(stderr, "[turbo-quant] %s: InnerQ auto-disabled (channels already balanced, max_ratio=%.3f)\n",
                __func__, max_ratio);
        innerq_enabled = 0;
        int zero = 0;
        cudaMemcpyToSymbol(d_innerq_calibrating, &zero, sizeof(int));
        return;
    }

    int zero = 0, one = 1;
    cudaMemcpyToSymbol(d_innerq_calibrating, &zero, sizeof(int));
    cudaMemcpyToSymbol(d_innerq_scale, scale, group_size * sizeof(float));
    cudaMemcpyToSymbol(d_innerq_scale_inv, scale_inv, group_size * sizeof(float));
    cudaDeviceSynchronize();
    cudaMemcpyToSymbol(d_innerq_active, &one, sizeof(int));

    innerq_enabled = 2;
    turbo_innerq_publish(scale_inv, group_size);

    fprintf(stderr, "[turbo-quant] %s: InnerQ finalized (%d tokens, max_ratio=%.3f, min_ratio=%.3f)\n",
            __func__, count, max_ratio, min_ratio);
}

void turbo_innerq_check_finalize(int group_size, int64_t ne00) {
    if (!innerq_initialized) {
        turbo_innerq_init();
    }
    if (innerq_enabled == 0) return;

    const bool multi_group_per_head = (group_size < 128);
    if (multi_group_per_head) {
        if (innerq_enabled == 1) {
            fprintf(stderr, "[turbo-quant WARN] %s: InnerQ disabled (ne00=%lld != group_size=%d, multi-group heads)\n",
                    __func__, (long long)ne00, group_size);
            innerq_enabled = 0;
            int zero = 0;
            cudaMemcpyToSymbol(d_innerq_calibrating, &zero, sizeof(int));
        }
        return;
    }

    if (innerq_enabled == 1) {
        int count = 0;
        cudaMemcpyFromSymbol(&count, d_innerq_count, sizeof(int));
        if (count >= innerq_target_tokens) {
            turbo_innerq_finalize(group_size);
        }
    }
}

bool turbo_innerq_is_active(void) {
    return innerq_enabled == 2;
}

// ─── CUDA kernel ──────────────────────────────────────────────────────────────
//
// Templated on direction and group_size (128 or 64).
// One block per group, group_size threads per block.
// direction: 0 = forward (signs1 → WHT → signs2), 1 = inverse (signs2 → WHT → signs1)
//
// When head_dim is not a multiple of group_size, only the full groups
// within each head are processed.  Tail elements are left unchanged (identity).
//
// Algorithm mirrors the CPU implementation in ggml-cpu/ops.cpp:
//   1. Apply s_first elementwise
//   2. Radix-2 Hadamard butterfly (log2(group_size) stages, in-place)
//   3. Normalize by 1/sqrt(group_size) and apply s_second elementwise
//
// InnerQ scale_inv: when non-null, applies per-channel inverse scaling for
// Q/V equalization. For forward (Q rotation): multiply BEFORE signs+WHT.
// For inverse (V un-rotation): multiply AFTER WHT+signs.

template <int direction, int group_size>
static __global__ void k_turbo_wht_f32(const float * __restrict__ src,
                                        float * __restrict__ dst,
                                        const float * __restrict__ scale_inv,
                                        int64_t n_groups,
                                        int64_t head_dim,
                                        int64_t groups_per_head) {
    static_assert(group_size == 128 || group_size == 64, "group_size must be 128 or 64");

    const int64_t g = blockIdx.x;
    if (g >= n_groups) return;

    const int t = threadIdx.x;  // 0 .. group_size-1

    // Map group index to position in the tensor:
    // each head has groups_per_head full groups, then a gap of tail elements.
    const int64_t head_idx     = g / groups_per_head;
    const int64_t grp_in_head  = g % groups_per_head;
    const int64_t base         = head_idx * head_dim + grp_in_head * group_size;

    __shared__ float x[group_size];

    // Load from global memory
    x[t] = src[base + t];
    __syncthreads();

    // InnerQ forward: apply scale_inv BEFORE signs+WHT (for Q pre-rotation)
    if (direction == 0 && scale_inv != nullptr) {
        x[t] *= scale_inv[t % group_size];
        __syncthreads();
    }

    // Apply first sign array
    if (group_size == 128) {
        x[t] *= (direction == 0) ? TURBO_WHT_SIGNS1[t] : TURBO_WHT_SIGNS2[t];
    } else {
        x[t] *= (direction == 0) ? TURBO_WHT_SIGNS1_64[t] : TURBO_WHT_SIGNS2_64[t];
    }
    __syncthreads();

    // WHT butterfly — log2(group_size) stages.
    // In stage h, threads where (t % (2h)) < h read x[t] and x[t+h],
    // then write x[t] = a+b and x[t+h] = a-b.  Each active thread
    // owns a disjoint pair, so no intra-stage conflicts exist.
#define WHT_STAGE(h) \
    if (t % (2*(h)) < (h)) { float a = x[t], b = x[t+(h)]; x[t] = a+b; x[t+(h)] = a-b; } \
    __syncthreads();

    WHT_STAGE(1)
    WHT_STAGE(2)
    WHT_STAGE(4)
    WHT_STAGE(8)
    WHT_STAGE(16)
    WHT_STAGE(32)
    if (group_size == 128) { WHT_STAGE(64) }
#undef WHT_STAGE

    // Normalize and apply second sign array, write to output
    constexpr float inv_sqrt = (group_size == 128) ? 0.08838834764831845f : 0.125f;
    float result;
    if (group_size == 128) {
        result = x[t] * inv_sqrt *
            ((direction == 0) ? TURBO_WHT_SIGNS2[t] : TURBO_WHT_SIGNS1[t]);
    } else {
        result = x[t] * inv_sqrt *
            ((direction == 0) ? TURBO_WHT_SIGNS2_64[t] : TURBO_WHT_SIGNS1_64[t]);
    }

    // InnerQ inverse: apply scale_inv AFTER WHT+signs (for V un-rotation)
    if (direction == 1 && scale_inv != nullptr) {
        result *= scale_inv[t % group_size];
    }

    dst[base + t] = result;
}

// ─── Simple copy kernel for tail elements (identity pass-through) ────────────

static __global__ void k_turbo_wht_copy_tail(const float * __restrict__ src,
                                              float * __restrict__ dst,
                                              int64_t n_heads,
                                              int64_t head_dim,
                                              int64_t tail_offset,
                                              int tail_size) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_heads * tail_size) return;

    const int64_t head_idx  = i / tail_size;
    const int64_t tail_elem = i % tail_size;
    const int64_t offset    = head_idx * head_dim + tail_offset + tail_elem;
    dst[offset] = src[offset];
}

// ─── Dispatch ─────────────────────────────────────────────────────────────────

void ggml_cuda_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    const ggml_tensor * scale_tensor = dst->src[1];  // InnerQ scale_inv (may be NULL)

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_is_contiguous(dst));

    int direction;
    int group_size;
    memcpy(&direction, dst->op_params + 0, sizeof(int));
    memcpy(&group_size, dst->op_params + sizeof(int), sizeof(int));

    const int64_t head_dim        = src->ne[0];
    const int64_t n_heads         = ggml_nelements(src) / head_dim;

    GGML_ASSERT(group_size == 64 || group_size == 128);
    const int64_t groups_per_head = head_dim / group_size;
    const int     tail_size       = (int)(head_dim % group_size);
    const int64_t n_groups        = groups_per_head * n_heads;

    const float * src_ptr = (const float *) src->data;
    float       * dst_ptr = (float       *) dst->data;
    const float * scale_inv_ptr = scale_tensor ? (const float *) scale_tensor->data : nullptr;

    cudaStream_t stream = ctx.stream();

    // Process full groups
    if (n_groups > 0) {
        dim3 blocks(n_groups);
        if (group_size == 128) {
            dim3 threads(128);
            if (direction == 0) {
                k_turbo_wht_f32<0, 128><<<blocks, threads, 0, stream>>>(src_ptr, dst_ptr, scale_inv_ptr, n_groups, head_dim, groups_per_head);
            } else {
                k_turbo_wht_f32<1, 128><<<blocks, threads, 0, stream>>>(src_ptr, dst_ptr, scale_inv_ptr, n_groups, head_dim, groups_per_head);
            }
        } else {
            dim3 threads(64);
            if (direction == 0) {
                k_turbo_wht_f32<0, 64><<<blocks, threads, 0, stream>>>(src_ptr, dst_ptr, scale_inv_ptr, n_groups, head_dim, groups_per_head);
            } else {
                k_turbo_wht_f32<1, 64><<<blocks, threads, 0, stream>>>(src_ptr, dst_ptr, scale_inv_ptr, n_groups, head_dim, groups_per_head);
            }
        }
    }

    // Pass through tail elements unchanged (no rotation)
    // Not needed for 64-aligned dims but kept for completeness
    if (tail_size > 0) {
        const int64_t total_tail = n_heads * tail_size;
        const int block_sz = 256;
        const int n_blocks = (int)((total_tail + block_sz - 1) / block_sz);
        k_turbo_wht_copy_tail<<<n_blocks, block_sz, 0, stream>>>(
            src_ptr, dst_ptr, n_heads, head_dim, groups_per_head * group_size, tail_size);
    }
}

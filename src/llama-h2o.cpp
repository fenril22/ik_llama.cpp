#include "llama-h2o.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <unordered_map>
#include <vector>

#include "llama-context.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

static std::unordered_map<int, std::vector<float>> g_scores_ema;

// Set H2O_VERBOSE=1 to enable eviction debug logging
static bool h2o_verbose() {
    static int v = -1;
    if (v < 0) { const char * e = getenv("H2O_VERBOSE"); v = (e && e[0] == '1') ? 1 : 0; }
    return v == 1;
}

static void dequant_row_to_f32(const void * row_data, float * out, ggml_type type, int head_dim) {
    if (type == GGML_TYPE_F32) {
        memcpy(out, row_data, head_dim * sizeof(float));
        return;
    }
    if (type == GGML_TYPE_F16) {
        const ggml_fp16_t * src = (const ggml_fp16_t *)row_data;
        for (int i = 0; i < head_dim; i++) {
            out[i] = ggml_fp16_to_fp32(src[i]);
        }
        return;
    }
    ggml_type_traits_t traits = ggml_internal_get_type_traits(type);
    if (traits.to_float) {
        traits.to_float(row_data, out, head_dim);
    } else {
        memset(out, 0, head_dim * sizeof(float));
    }
}

static float dot_f32(const float * a, const float * b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

static bool cell_matches_seq(const llama_kv_cell & cell, llama_seq_id seq_id) {
    if (cell.pos < 0) return false;
    if (seq_id < 0) return !cell.is_empty();
    return cell.has_seq_id(seq_id);
}

// Score all KV entries and evict the lowest-scoring contiguous block.
// Contiguous eviction ensures kv_cache_find_slot can find free slots.
static int h2o_do_evict(struct llama_context * ctx, const h2o_params & params,
                        int n_to_evict, llama_seq_id seq_id, const char * caller = "?") {
    if (n_to_evict <= 0) return 0;

    auto & cache = ctx->kv_self;
    const auto & hparams = ctx->model.hparams;

    const int kv_size    = (int)cache.size;

    int score_layer = 0;
    if (cache.hybrid) {
        for (uint32_t il = 0; il < hparams.n_layer; il++) {
            if (!hparams.recurrent_layer_arr[il]) {
                score_layer = (int)il;
                break;
            }
        }
    }

    // Use score_layer's actual dimensions, not layer 0's
    const int n_head_kv  = (int)hparams.n_head_kv(score_layer);
    const int head_dim   = (int)hparams.n_embd_head_k(score_layer);

    // Find the latest KV cell
    int latest_cell = -1;
    llama_pos latest_pos = -1;
    for (int i = 0; i < kv_size; i++) {
        if (!cell_matches_seq(cache.cells[i], seq_id)) continue;
        if (cache.cells[i].pos > latest_pos) {
            latest_pos = cache.cells[i].pos;
            latest_cell = i;
        }
    }
    if (latest_cell < 0) return 0;

    struct ggml_tensor * k_tensor = cache.k_l[score_layer];
    if (!k_tensor || !k_tensor->data) return 0;

    const ggml_type type_k = k_tensor->type;
    const size_t row_stride = k_tensor->nb[1];

    const size_t total_bytes = ggml_nbytes(k_tensor);
    std::vector<uint8_t> k_host(total_bytes);
    ggml_backend_tensor_get(k_tensor, k_host.data(), 0, total_bytes);

    auto k_row_ptr = [&](int h, int i) -> const void * {
        size_t row_idx = (size_t)h * kv_size + (size_t)i;
        return k_host.data() + row_idx * row_stride;
    };

    std::vector<float> k_latest(n_head_kv * head_dim);
    for (int h = 0; h < n_head_kv; h++) {
        dequant_row_to_f32(k_row_ptr(h, latest_cell), k_latest.data() + h * head_dim, type_k, head_dim);
    }

    // Compute per-cell scores
    std::vector<float> scores(kv_size, -1e30f); // non-matching cells get very low score
    std::vector<float> k_cur(head_dim);

    for (int i = 0; i < kv_size; i++) {
        if (!cell_matches_seq(cache.cells[i], seq_id)) continue;
        if (cache.cells[i].pos < params.kv_sink) { scores[i] = 1e30f; continue; } // sink: never evict
        if (i == latest_cell)                     { scores[i] = 1e30f; continue; } // latest: never evict

        float score = 0.0f;
        for (int h = 0; h < n_head_kv; h++) {
            dequant_row_to_f32(k_row_ptr(h, i), k_cur.data(), type_k, head_dim);
            score += dot_f32(k_latest.data() + h * head_dim, k_cur.data(), head_dim);
        }
        scores[i] = score / n_head_kv;
    }

    // EMA update
    int ema_key = (seq_id >= 0) ? seq_id : -1;
    auto & ema = g_scores_ema[ema_key];
    if ((int)ema.size() < kv_size) {
        ema.resize(kv_size, 0.0f);
    }
    const float alpha = 0.3f;
    for (int i = 0; i < kv_size; i++) {
        if (!cell_matches_seq(cache.cells[i], seq_id)) continue;
        ema[i] = alpha * scores[i] + (1.0f - alpha) * ema[i];
    }

    // Find the best contiguous block of n_to_evict cells to evict.
    // Use sliding window over cell indices with lowest sum of EMA scores.
    // Cells that are sinks, latest, or empty get score=1e30 so they won't be chosen.
    float window_sum = 0.0f;
    for (int i = 0; i < n_to_evict && i < kv_size; i++) {
        window_sum += ema[i];
    }
    float best_sum = window_sum;
    int best_start = 0;

    for (int i = 1; i + n_to_evict <= kv_size; i++) {
        window_sum -= ema[i - 1];
        window_sum += ema[i + n_to_evict - 1];
        if (window_sum < best_sum) {
            best_sum = window_sum;
            best_start = i;
        }
    }

    // Evict the contiguous block using pos range
    llama_seq_id rm_seq = (seq_id >= 0) ? seq_id : -1;
    llama_pos pos_min = cache.cells[best_start].pos;
    llama_pos pos_max = cache.cells[best_start + n_to_evict - 1].pos;

    // Range removal covers all positions in the block
    if (pos_min >= 0 && pos_max >= pos_min) {
        llama_kv_cache_seq_rm(ctx, rm_seq, pos_min, pos_max + 1);
    }

    // Reset cache head to the evicted region so find_slot finds it immediately
    cache.head = best_start;

    int actual_evicted = 0;
    for (int i = best_start; i < best_start + n_to_evict && i < kv_size; i++) {
        if (cache.cells[i].pos < 0) actual_evicted++;
    }

    if (h2o_verbose()) fprintf(stderr, "h2o[%s]: seq=%d evicted %d cells [%d..%d] pos=[%d..%d] (kv_used=%d)\n",
            caller, (int)seq_id, actual_evicted, best_start, best_start + n_to_evict - 1,
            (int)pos_min, (int)pos_max, llama_get_kv_cache_used_cells(ctx));

    return actual_evicted;
}

int h2o_maybe_evict(struct llama_context * ctx, const h2o_params & params, int n_past,
                    llama_seq_id seq_id, bool force) {
    if (params.kv_budget <= 0) return 0;
    if (n_past <= 0) return 0;
    if (!force && (n_past % params.kv_evict_interval != 0)) return 0;

    const int kv_used = llama_get_kv_cache_used_cells(ctx);
    if (kv_used <= (int)(params.kv_budget * 0.9)) return 0;

    int n_to_evict;
    if (seq_id >= 0) {
        auto & cache = ctx->kv_self;
        int seq_count = 0;
        for (uint32_t i = 0; i < cache.size; i++) {
            if (cell_matches_seq(cache.cells[i], seq_id)) seq_count++;
        }
        n_to_evict = seq_count - params.kv_budget;
    } else {
        const int target = (int)(params.kv_budget * 0.7);
        n_to_evict = kv_used - target;
    }

    return h2o_do_evict(ctx, params, n_to_evict, seq_id, "proactive");
}

int h2o_ensure_budget(struct llama_context * ctx, const h2o_params & params,
                      int n_needed, llama_seq_id seq_id) {
    if (params.kv_budget <= 0) return 0;

    auto & cache = ctx->kv_self;
    const int kv_size = (int)cache.size;
    const int kv_used = llama_get_kv_cache_used_cells(ctx);
    const int kv_free = kv_size - kv_used;

    if (kv_free >= n_needed) return 0;

    // Evict enough for n_needed contiguous free slots
    int n_to_evict = n_needed;

    return h2o_do_evict(ctx, params, n_to_evict, seq_id, "reactive");
}

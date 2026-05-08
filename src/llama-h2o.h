#pragma once

#include "llama.h"

struct h2o_params {
    int32_t kv_budget         = -1;   // max KV cache entries (-1 = unlimited)
    int32_t kv_sink           = 2048; // first N tokens protected from eviction
    int32_t kv_evict_interval = 256;  // evict every N decode steps
};

// H2O KV cache eviction (CPU K-K dot product + seq_rm)
//
// Call after each decode step. Checks if eviction is needed and performs it.
// seq_id: sequence to evict from (-1 = all sequences, for single-sequence mode)
// n_past: current decode step count (used for interval check)
// force: if true, skip interval check and evict immediately when over budget
// Returns the number of KV entries evicted (0 if none).
int h2o_maybe_evict(struct llama_context * ctx, const h2o_params & params, int n_past,
                    llama_seq_id seq_id = -1, bool force = false);

// Ensure KV cache has at least n_needed free slots by evicting if necessary.
// Call before llama_decode during prefill to make room for the next batch.
// Returns the number of KV entries evicted (0 if none).
int h2o_ensure_budget(struct llama_context * ctx, const h2o_params & params,
                      int n_needed, llama_seq_id seq_id = -1);

// Notify H2O that a sequence's KV cache entries have been fully evicted (all
// tokens removed, i.e. llama_kv_cache_seq_rm with p0=-1/p1=-1, or a full clear).
// Must be called from any site that performs a top-level full-sequence eviction
// so that stale EMA scores are not carried into the next request.
// Pass seq_id < 0 to clear EMA scores for all sequences.
void h2o_on_seq_rm(llama_seq_id seq_id);

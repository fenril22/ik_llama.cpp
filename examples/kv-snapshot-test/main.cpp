#include "common.h"
#include "llama.h"
#include "llama-h2o.h"
#include "llama-snapshot-store.h"

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

// Built-in filler prompt used when the user's --prompt is too short (<50 tokens).
static const char * FILLER_PROMPT =
    "The quick brown fox jumps over the lazy dog. "
    "In a hole in the ground there lived a hobbit. "
    "It was the best of times, it was the worst of times. "
    "To be or not to be, that is the question. "
    "Call me Ishmael. Some years ago, never mind how long precisely, "
    "having little or no money in my purse, and nothing particular to "
    "interest me on shore, I thought I would sail about a little and "
    "see the watery part of the world. "
    "It is a truth universally acknowledged, that a single man in possession "
    "of a good fortune, must be in want of a wife. "
    "Happy families are all alike; every unhappy family is unhappy in its own way. "
    "It was a bright cold day in April, and the clocks were striking thirteen. "
    "The sky above the port was the color of television, tuned to a dead channel.";

static const char * AGENT_B_PROMPT =
    "Write a short poem about the sea and its endless waves.";

static double elapsed_ms(
    std::chrono::high_resolution_clock::time_point t0,
    std::chrono::high_resolution_clock::time_point t1)
{
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main(int argc, char ** argv) {
    // Parse CLI args (handles --model, --kv-budget, --kv-sink,
    // --kv-evict-interval, --kv-snapshot-max-mem, --prompt, etc.)
    gpt_params params;
    if (!gpt_params_parse(argc, argv, params)) {
        gpt_params_print_usage(argc, argv, params);
        return 1;
    }

    // Init backend and model/context.
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_init_result llama_init = llama_init_from_gpt_params(params);
    llama_model   * model = llama_init.model;
    llama_context * ctx   = llama_init.context;
    if (!model || !ctx) {
        fprintf(stderr, "Failed to load model/context\n");
        return 1;
    }

    // Build h2o_params from common params.
    h2o_params h2o;
    h2o.kv_budget         = params.kv_budget;
    h2o.kv_sink           = params.kv_sink;
    h2o.kv_evict_interval = params.kv_evict_interval;

    // Init snapshot store.
    kv_snapshot_store store;
    if (params.kv_snapshot_max_mem > 0) {
        store.set_max_memory((size_t)params.kv_snapshot_max_mem * 1024UL * 1024UL);
    }
    fprintf(stderr, "Snapshot store: max_mem=%zu MiB (%s)\n",
            (size_t)params.kv_snapshot_max_mem,
            params.kv_snapshot_max_mem > 0 ? "enabled" : "disabled");

    const bool add_bos = llama_vocab_get_add_bos(llama_model_get_vocab(model));
    const int  n_batch = llama_n_batch(ctx);

    // Choose prompt for Agent A — use built-in filler when user prompt is short.
    const std::string prompt_a = (!params.prompt.empty() &&
                                  (int)params.prompt.size() > 50)
                                 ? params.prompt
                                 : FILLER_PROMPT;

    // -----------------------------------------------------------------------
    // Phase A — Agent A: prefill + save snapshot
    // -----------------------------------------------------------------------
    fprintf(stderr, "\n=== Phase A: Agent A prefill + snapshot save ===\n");

    std::vector<llama_token> tokens_a =
        ::common_tokenize(model, prompt_a, add_bos);
    fprintf(stderr, "Agent A prompt tokens: %d\n", (int)tokens_a.size());

    int n_past_a = 0;
    for (int i = 0; i < (int)tokens_a.size(); i += n_batch) {
        int n_eval = std::min(n_batch, (int)tokens_a.size() - i);
        if (h2o.kv_budget > 0) {
            h2o_ensure_budget(ctx, h2o, n_eval, /*seq_id=*/0);
        }
        if (llama_decode(ctx, llama_batch_get_one(&tokens_a[i], n_eval, n_past_a, 0))) {
            fprintf(stderr, "Agent A: decode failed at position %d\n", n_past_a);
            return 1;
        }
        n_past_a += n_eval;
    }
    fprintf(stderr, "Agent A prefill done: n_past=%d, kv_used=%d\n",
            n_past_a, llama_get_kv_cache_used_cells(ctx));

    // Measure snapshot size before saving.
    size_t snap_size_est = llama_state_seq_get_size(ctx, /*seq_id=*/0, 0);
    fprintf(stderr, "Snapshot estimated size: %zu bytes (%.1f MiB)\n",
            snap_size_est, snap_size_est / (1024.0 * 1024.0));

    auto t_save0 = std::chrono::high_resolution_clock::now();
    store.save(ctx, /*seq_id=*/0, tokens_a);
    auto t_save1 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "Snapshot saved in %.1f ms  (store count=%d, total_mem=%zu bytes)\n",
            elapsed_ms(t_save0, t_save1), store.count(), store.total_memory());

    // -----------------------------------------------------------------------
    // Phase B — Agent B "steals" the KV slot
    // -----------------------------------------------------------------------
    fprintf(stderr, "\n=== Phase B: Agent B steals the slot ===\n");

    llama_kv_cache_clear(ctx);
    h2o_on_seq_rm(-1);

    std::vector<llama_token> tokens_b =
        ::common_tokenize(model, AGENT_B_PROMPT, add_bos);
    fprintf(stderr, "Agent B prompt tokens: %d\n", (int)tokens_b.size());

    int n_past_b = 0;
    for (int i = 0; i < (int)tokens_b.size(); i += n_batch) {
        int n_eval = std::min(n_batch, (int)tokens_b.size() - i);
        if (llama_decode(ctx, llama_batch_get_one(&tokens_b[i], n_eval, n_past_b, 0))) {
            fprintf(stderr, "Agent B: decode failed at position %d\n", n_past_b);
            return 1;
        }
        n_past_b += n_eval;
    }
    fprintf(stderr, "Agent B prefill done (slot stolen): kv_used=%d\n",
            llama_get_kv_cache_used_cells(ctx));

    // -----------------------------------------------------------------------
    // Phase C — Agent A comes back, restores from snapshot
    // -----------------------------------------------------------------------
    fprintf(stderr, "\n=== Phase C: Agent A restore + delta prefill ===\n");

    // Simulate slot eviction: clear the KV before Agent A tries to resume.
    llama_kv_cache_clear(ctx);
    h2o_on_seq_rm(-1);
    fprintf(stderr, "KV cleared (slot eviction simulated): kv_used=%d\n",
            llama_get_kv_cache_used_cells(ctx));

    // Build tokens_a2 = original Agent A tokens + a few continuation tokens.
    const std::string continuation = " Furthermore, the implications of this are profound.";
    std::vector<llama_token> extra_tokens =
        ::common_tokenize(model, continuation, /*add_special=*/false);
    std::vector<llama_token> tokens_a2 = tokens_a;
    tokens_a2.insert(tokens_a2.end(), extra_tokens.begin(), extra_tokens.end());
    fprintf(stderr, "Agent A continuation tokens: base=%d  extra=%d  total=%d\n",
            (int)tokens_a.size(), (int)extra_tokens.size(), (int)tokens_a2.size());

    auto t_restore0 = std::chrono::high_resolution_clock::now();
    int restored = store.restore(ctx, /*seq_id=*/0, tokens_a2);
    auto t_restore1 = std::chrono::high_resolution_clock::now();

    fprintf(stderr, "Restore returned: %d tokens (kv_used=%d)  in %.1f ms\n",
            restored, llama_get_kv_cache_used_cells(ctx),
            elapsed_ms(t_restore0, t_restore1));

    if (restored > 0) {
        // Only prefill the delta (tokens not already in the KV cache).
        int delta_start = restored;
        int delta_count = (int)tokens_a2.size() - delta_start;
        fprintf(stderr, "Delta prefill: %d tokens (positions %d..%d)\n",
                delta_count, delta_start, (int)tokens_a2.size() - 1);

        auto t_delta0 = std::chrono::high_resolution_clock::now();
        int n_past_c = restored;
        for (int i = delta_start; i < (int)tokens_a2.size(); i += n_batch) {
            int n_eval = std::min(n_batch, (int)tokens_a2.size() - i);
            if (h2o.kv_budget > 0) {
                h2o_ensure_budget(ctx, h2o, n_eval, /*seq_id=*/0);
            }
            if (llama_decode(ctx,
                    llama_batch_get_one(&tokens_a2[i], n_eval, n_past_c, 0))) {
                fprintf(stderr, "Agent A delta: decode failed at position %d\n", n_past_c);
                return 1;
            }
            n_past_c += n_eval;
        }
        auto t_delta1 = std::chrono::high_resolution_clock::now();

        fprintf(stderr, "Delta prefill done: n_past=%d, kv_used=%d  in %.1f ms\n",
                n_past_c, llama_get_kv_cache_used_cells(ctx),
                elapsed_ms(t_delta0, t_delta1));
    } else {
        fprintf(stderr, "No snapshot restored (store disabled or no match)\n");
    }

    // -----------------------------------------------------------------------
    // Stats
    // -----------------------------------------------------------------------
    fprintf(stderr, "\n=== Snapshot Store Stats ===\n");
    fprintf(stderr, "  Snapshots stored : %d\n",   store.count());
    fprintf(stderr, "  Total memory used: %zu bytes (%.1f MiB)\n",
            store.total_memory(), store.total_memory() / (1024.0 * 1024.0));
    fprintf(stderr, "  Max memory limit : %zu MiB\n",
            store.get_max_memory() / (1024UL * 1024UL));

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    fprintf(stderr, "\nDone.\n");
    return 0;
}

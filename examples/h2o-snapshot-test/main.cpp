#include "common.h"
#include "llama.h"
#include "llama-h2o.h"
#include <cstdio>
#include <vector>
#include <chrono>

int main(int argc, char ** argv) {
    gpt_params params;
    if (!gpt_params_parse(argc, argv, params)) {
        gpt_params_print_usage(argc, argv, params);
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_init_result llama_init = llama_init_from_gpt_params(params);
    llama_model * model = llama_init.model;
    llama_context * ctx = llama_init.context;
    if (!model || !ctx) {
        fprintf(stderr, "Failed to load model/context\n");
        return 1;
    }

    // Tokenize prompt
    const bool add_bos = llama_vocab_get_add_bos(llama_model_get_vocab(model));
    std::vector<llama_token> tokens = ::common_tokenize(model, params.prompt, add_bos);
    int n_tokens = (int)tokens.size();
    fprintf(stderr, "Prompt tokens: %d\n", n_tokens);

    // Prefill in batches with H2O eviction
    h2o_params h2o;
    h2o.kv_budget         = params.kv_budget;
    h2o.kv_sink           = params.kv_sink;
    h2o.kv_evict_interval = params.kv_evict_interval;

    int n_past = 0;
    int n_batch = llama_n_batch(ctx);
    for (int i = 0; i < n_tokens; i += n_batch) {
        int n_eval = std::min(n_batch, n_tokens - i);
        if (h2o.kv_budget > 0) {
            h2o_ensure_budget(ctx, h2o, n_eval);
        }
        if (llama_decode(ctx, llama_batch_get_one(&tokens[i], n_eval, n_past, 0))) {
            fprintf(stderr, "Failed to decode at n_past=%d\n", n_past);
            return 1;
        }
        n_past += n_eval;
    }
    fprintf(stderr, "Prefill done: n_past=%d, kv_used=%d\n", n_past, llama_get_kv_cache_used_cells(ctx));

    // Measure snapshot size
    size_t snap_size = llama_state_seq_get_size(ctx, 0, 0);
    fprintf(stderr, "\n=== KV Snapshot ===\n");
    fprintf(stderr, "Size: %zu bytes (%.1f MB)\n", snap_size, snap_size / (1024.0 * 1024.0));

    // Save snapshot
    std::vector<uint8_t> snap_buf(snap_size);
    auto t0 = std::chrono::high_resolution_clock::now();
    size_t written = llama_state_seq_get_data(ctx, snap_buf.data(), snap_buf.size(), 0, 0);
    auto t1 = std::chrono::high_resolution_clock::now();
    double save_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fprintf(stderr, "Save:    %zu bytes in %.1f ms (%.1f GB/s)\n",
            written, save_ms, written / save_ms / 1e6);

    // Clear KV and restore
    llama_kv_cache_clear(ctx);
    fprintf(stderr, "KV cleared: kv_used=%d\n", llama_get_kv_cache_used_cells(ctx));

    auto t2 = std::chrono::high_resolution_clock::now();
    size_t restored = llama_state_seq_set_data(ctx, snap_buf.data(), snap_buf.size(), 0, 0);
    auto t3 = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    fprintf(stderr, "Restore: %zu bytes in %.1f ms (%.1f GB/s)\n",
            restored, load_ms, restored / load_ms / 1e6);
    fprintf(stderr, "KV restored: kv_used=%d\n", llama_get_kv_cache_used_cells(ctx));

    // Full state size for comparison
    size_t full_size = llama_state_get_size(ctx);
    fprintf(stderr, "\nFull state: %.1f MB\n", full_size / (1024.0 * 1024.0));

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}

# H2O KV Cache Eviction: Future Work

## GPU-side Scoring Kernel

### Background

The current H2O eviction implementation (`src/llama-h2o.cpp`, `h2o_do_evict`) performs
KV cache scoring on the CPU:

1. Transfers the entire K-cache tensor for `score_layer` from GPU to CPU (19 MB via PCIe, ~1.5 ms)
2. Dequantizes each K-row from turbo3c format on CPU (~5–8 ms, dequant-bound)
3. Computes dot-product scores against the latest token's K-vector
4. Updates per-cell EMA scores and finds the lowest-scoring contiguous window

Per eviction call: ~6.5–9.5 ms total. GPU-side equivalent: ~50–100 µs (65–150× faster).

### Why it matters (future configurations)

With the current setup (2 proactive evictions per run), the actual impact is ~18 ms per
6-minute run (0.005%). However, in configurations with:
- Shorter KV budgets relative to context size
- Higher reactive eviction rates
- Multiple concurrent sequences

...the per-call cost accumulates and GPU-side scoring becomes worthwhile.

### Implementation notes

**Template code available:**
- `ggml/src/ggml-cuda/dmmv.cu` — dequantize-row + dot-with-float-vector → scalar per row
  (structurally identical to the H2O scoring kernel)
- `ggml/src/ggml-cuda/turbo-quant.cuh` — GPU-side turbo3c dequant primitives
- `ggml/src/ggml-cuda/argsort.cu` — `k_topk_sum` (sliding window minimum sum variant)
- `ggml/src/ggml-cuda/common.cuh` — `warp_reduce_sum` for dot product accumulation

**Design considerations:**
- EMA buffer must be persistent device memory tied to `llama_context` lifetime, sized to `kv_size`
- `best_start` (4 bytes) still needs CPU readback to call `llama_kv_cache_seq_rm`; the 19 MB transfer is eliminated but one small D2H copy remains
- Recommended validation: dual-path mode (CPU + GPU run in parallel, assert same `best_start`) before removing CPU path
- Integration points: `llama_context` alloc/free hooks for device buffer, both `h2o_maybe_evict` and `h2o_ensure_budget` call sites, hybrid-model `score_layer` path

**Estimated new CUDA:** ~300–400 lines including integration.

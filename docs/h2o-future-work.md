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

---

# Performance Bottleneck Analysis (2026-05-06)

## Environment

- GPU: RTX 3070 (8 GB VRAM, 448 GB/s bandwidth)
- CPU: AMD Ryzen 9 9950X (16C/32T, 24 logical cores available)
- Model: Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf (MoE, Q3_K_XL)
- Config: `-c 204800 --kv-budget 100000 --n-cpu-moe 31 -ngl 99`

## Observed Performance

| Phase | Speed |
|---|---|
| Prefill | ~644 tok/s |
| Decode | ~33 tok/s |

## Bottleneck: CPU MoE (not GPU memory bandwidth)

Contrary to initial expectation, decode is **not** GPU memory-bandwidth-limited.

Measured during decode:
- GPU SM utilization: **71–72%** (headroom exists)
- GPU VRAM bandwidth utilization: **60%** (headroom exists)
- CPU utilization: **~47%** (~11 threads, significant idle capacity)

The bottleneck is **CPU MoE expert computation**: with `--n-cpu-moe 31`, each decode
step dispatches 31 MoE expert layers to CPU sequentially, creating a CPU↔GPU sync point
per MoE block per token.

## Thread Count Sweep (`-t` parameter)

Tested `-t 12` (baseline) through `-t 24` to find the optimal decode thread count:

| `-t` | Prefill (tok/s) | Decode (tok/s) |
|---|---|---|
| 12 (baseline) | 643 | 33.0 |
| 16 | 644 | 33.1 |
| 20 | 644 | 32.3 |
| 24 | 644 | 23.6 ← regression |

**Findings:**
- Prefill is GPU-bound and unaffected by `-t`
- Decode peaks at `-t 16` but improvement over `-t 12` is marginal (+0.3%)
- `-t 24` causes severe regression due to contention with `-tb 24` (prefill threads)
- **`-t 12` is effectively optimal** — decode speed is not thread-count-limited

## Why decode cannot be improved further (current hardware)

Decode at 33 tok/s with CPU MoE is constrained by the serial CPU↔GPU dispatch per MoE
layer. To improve:

1. **Reduce `--n-cpu-moe`** (move more MoE experts to GPU) — VRAM is at ~7.2/8 GB,
   very limited headroom (~880 MB free during decode with `--n-cpu-moe 31`)
2. **GPU upgrade** (e.g. RTX 4090: 24 GB VRAM, 1008 GB/s) — would allow all MoE
   experts on GPU, eliminating CPU↔GPU sync; decode speed would increase 2–3×
3. **Quantize further** (Q2_K etc.) — reduces model size, allows more experts on GPU,
   at quality cost

## PCIe Anomaly

PCIe was observed running at **Gen 1 (2.5 GT/s)** instead of expected Gen 4 (16 GT/s).
At current transfer volumes (~300 MB/s peak), this is not yet rate-limiting, but should
be investigated (BIOS power-saving setting or hardware issue). If batch sizes increase or
more data is transferred, Gen 1 will become a bottleneck.

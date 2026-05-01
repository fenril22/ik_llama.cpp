# RotorQuant KV Cache Compression

ik_llama.cpp supports RotorQuant-style KV cache compression via two quantization schemes:
**PlanarQuant** (2D Givens rotation) and **IsoQuant** (quaternion 4D rotation).
These reduce KV cache VRAM by ~75–80% at the cost of minor generation speed reduction.

## Quantization Types

| Type | Bits | Rotation | VRAM vs f16 |
|------|------|----------|-------------|
| `planar3` | 3-bit | 2D Givens (per pair) | −75% |
| `planar4` | 4-bit | 2D Givens (per pair) | −50% |
| `iso3` | 3-bit | Quaternion 4D | −75% |
| `iso4` | 4-bit | Quaternion 4D | −50% |

**Recommended: `iso3`** — best VRAM reduction with acceptable quality.

## Usage

```bash
./build/bin/llama-cli \
  -m models/your-model.gguf \
  -ctk iso3 -ctv iso3 \
  -ngl 99 \
  --flash-attn 1
```

For MoE models with CPU offload (e.g. Qwen3.6-35B-A3B on 8GB GPU):

```bash
GGML_CUDA_NO_PINNED=1 ./build/bin/llama-cli \
  -m models/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf \
  -ctk iso3 -ctv iso3 \
  -ngl 99 \
  --n-cpu-moe 31 \
  -b 512 -ub 512 \
  --flash-attn 1 \
  -t 12 \
  --run-time-repack
```

## Benchmark Results

**Model**: Qwen3.6-35B-A3B Q3\_K\_XL  
**GPU**: NVIDIA GeForce RTX 3070 (8GB VRAM)  
**Context**: 512 prompt + 128 generation

| type\_k | type\_v | pp512 (t/s) | tg128 (t/s) |
|---------|---------|-------------|-------------|
| f16 | f16 | 617 ± 313 | **61.9** |
| f16 | iso3 | 751 ± 9 | 57.6 |
| iso3 | f16 | 752 ± 11 | 57.5 |
| iso3 | iso3 | 751 ± 6 | **55.0** |

**Key observations:**
- **Prefill +22% faster** with iso3 — VRAM pressure reduction eliminates swap/stall (±313→±6 variance drop)
- **Generation −11%** — dequantize overhead (iso3→f16 per attention step)
- f16/f16 showed extreme variance (±312) indicating VRAM overflow on 8GB GPU at this context length

## VRAM Savings

For a model with head\_dim=128, n\_heads\_kv=8, n\_layers=64:

| type | bytes/token/layer | 4096-token context |
|------|-------------------|--------------------|
| f16 | 2048 B | ~500 MiB |
| iso3 | 512 B | ~125 MiB |

In practice with Qwen3.6-35B-A3B:
- f16 KV: ~5120 MiB → **OOM on 8GB GPU at long context**
- iso3 KV: ~1000 MiB → fits comfortably

## Implementation Notes

- CUDA encode kernels: F16/F32 → planar3/planar4/iso3/iso4 (one thread per block of 128 elements)
- CUDA decode kernels: planar3/planar4/iso3/iso4 → F16 (for fattn-mma path)
- CPU fallback: fully implemented for all four types
- Rotation constants are fixed (seed=42) and identical between CPU and CUDA
- Thread-safe initialization via `pthread_once` (CPU) and `std::call_once` (CUDA)
- CUDA graph support: disabled for CPY(→quant) ops (returns nullptr from `ggml_cuda_cpy_fn`)

## llama-bench

```bash
./build/bin/llama-bench \
  -m models/your-model.gguf \
  -ctk f16,iso3 -ctv f16,iso3 \
  -p 512 -n 128
```

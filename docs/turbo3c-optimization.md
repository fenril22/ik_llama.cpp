# TURBO3C KV Cache Optimization

## Overview

TURBO3C (`GGML_TYPE_TURBO3C_0`, type=49) is a 3-bit KV cache quantization type
based on Walsh-Hadamard Transform (WHT) + integer centroid quantization.
It is designed for DP4A (INT8 dot product) acceleration on NVIDIA GPUs.

Compared to the original turbo3, turbo3c uses integer centroids `[-8,-5,-3,-1,1,3,5,8]`
(scale=0.023568) instead of Lloyd-Max optimal centroids, enabling GPU integer arithmetic.

## Block Structure

```
block_turbo3c_0 {
    ggml_half norm;              // 2 bytes: block normalization factor
    uint8_t   qs[QK/4];         // 32 bytes: lower 2-bit of 3-bit index (4 per byte)
    uint8_t   signs[QK/8];      // 16 bytes: upper 1-bit of 3-bit index (8 per byte)
}
// Total: 50 bytes per 128 values = 3.125 bits/value
```

## GPU Optimization Details

### Decode Path (ne[1]==1): MMA via F16 Conversion

For token generation (decode), turbo3c bypasses the fattn-vec kernel and uses the
MMA (Matrix Multiply-Accumulate) TensorCore path instead:

1. turbo3c data is converted to F16 on-the-fly via `dequantize_row_turbo3c_0_cuda`
2. F16 data is processed by the MMA attention kernel using TensorCores
3. This trades a temporary F16 buffer (~134 MiB/layer) for significantly faster computation

The F16 conversion kernel uses **LUT-free decoding**: instead of `__constant__` memory
table lookups, centroid values are packed into register constants:

```cuda
constexpr int NEG_PACKED = (int)0xFFFDFBF8;  // [-8,-5,-3,-1] as int8x4
constexpr int POS_PACKED = (int)0x08050301;   // [+1,+3,+5,+8] as int8x4
int8_t val = (int8_t)((hi1 ? POS_PACKED : NEG_PACKED) >> (low2 * 8));
```

This eliminates `__constant__` memory latency (L2 cache, ~tens of cycles) and replaces
it with register shift operations (~1 cycle).

### Prefill Path (ne[1]>1): MMA via F16 Conversion

Same as decode - turbo3c data is converted to F16, then processed by TensorCore MMA.
This path was already in place before the turbo3c optimizations.

### DP4A Path (alternative decode, not default)

A DP4A-based decode path also exists in `fattn-vec-common.cuh` using `turbo3c_int4_packed`
for batch decoding of 4 int8 values. This path is currently not the default for turbo3c
(MMA is faster at long contexts) but remains available for other turbo types.

## Performance Benchmarks

Model: Qwen3.6-35B-A3B MoE (IQ3_S), RTX 3070 8GB, --n-cpu-moe 30 --flash-attn 1

### Decode Speed (128k context, t128.txt, 20 tokens)

| KV Type              | KV Size  | Decode (tok/s) | vs q4_0 |
|----------------------|---------:|:--------------:|:-------:|
| q4_0                 |  720 MiB |     30.85      |  100%   |
| q4_1                 |  800 MiB |     30.66      |   99%   |
| turbo3c (MMA+LUT-free) |  500 MiB |     27.42      |   89%   |
| t3c + q4_1 F/L 10   |  650 MiB |     29.15      |   95%   |
| t3c + q4_1 F/L 5    |  590 MiB |     28.16      |   91%   |
| turbo3 (vec, baseline) |  500 MiB |     11.25      |   36%   |

### Perplexity (t128.txt)

| KV Type              | 8k PPL  | 32k PPL | vs f16 @32k |
|----------------------|:-------:|:-------:|:-----------:|
| f16                  | 1.2641  | 2.1982  |      —      |
| q4_0                 | 1.2828  | 2.2369  |    +1.8%    |
| q4_1                 | 1.3276  | 2.0560  |    -6.5%    |
| turbo3c              | 1.3426  | 2.7694  |   +26.0%    |
| t3c + q4_1 F/L 5    | 1.2663  | 2.4322  |   +10.6%    |
| **t3c + q4_1 F/L 10** | **1.2400** | **2.0043** | **-8.8%** |

## Recommended Configuration

For the best balance of memory savings, speed, and precision:

```bash
# Recommended: turbo3c + q4_1 first/last 10 layers
-ctk turbo3c -ctv turbo3c \
-ctk-first q4_1,10 -ctv-first q4_1,10 \
-ctk-last q4_1,10 -ctv-last q4_1,10
```

This gives:
- **KV memory**: 650 MiB at 128k (q4_0 -10%, turbo3c +30%)
- **Decode speed**: 29.15 tok/s (q4_0 95%)
- **Precision**: Better than f16 at both 8k and 32k contexts

### Layer Distribution (40-layer model)

| Layers   | KV Type  | Purpose                     |
|----------|----------|-----------------------------|
| 0-9      | q4_1     | High precision (input-sensitive) |
| 10-29    | turbo3c  | Memory savings (WHT rotation)   |
| 30-39    | q4_1     | High precision (output-sensitive) |

### Why 50% q4_1?

- **25% (F/L 5)**: Good at 8k but degrades +10% at 32k
- **50% (F/L 10)**: Optimal - beats f16 at both 8k and 32k
- **100% q4_1**: Worse at 8k (+5%), more memory, no benefit

## Key Implementation Notes

### Files Modified (from base ik_llama.cpp)

| File | Change |
|------|--------|
| `ggml-backend.cpp` | CPU backend rejects TURBO CPY ops (SIGSEGV fix) |
| `turbo-quant.cuh` | LUT-free `turbo3c_int_from_bits`, `turbo3c_int4_packed`, `turbo3c_dequant_element` |
| `fattn-vec-common.cuh` | DP4A dot product with LUT-free batch decode |
| `fattn-vec-f16.cuh` | turbo3c uses Q_q8_1 quantization path |
| `fattn.cu` | turbo3c decode routed to MMA path (not fattn-vec) |

### Why not QJL residual correction?

The TurboQuant paper (arXiv 2504.19874) proposes a QJL (Quantized Johnson-Lindenstrauss)
1-bit residual correction stage. However:

1. QJL adds +1 bit/val memory overhead (3.125 -> 4.125 bit/val)
2. QJL restores mathematical unbiasedness of dot products, but softmax amplifies QJL's
   variance more than the bias correction helps
3. The paper's own implementations disable QJL at 3+ bits
4. The mixed layer strategy (turbo3c + q4_1) achieves better results without QJL

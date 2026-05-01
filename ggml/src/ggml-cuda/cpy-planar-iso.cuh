#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

void ggml_cuda_cpy_f16_planar3(const char * src, char * dst, int64_t ne, cudaStream_t stream);
void ggml_cuda_cpy_f16_planar4(const char * src, char * dst, int64_t ne, cudaStream_t stream);
void ggml_cuda_cpy_f16_iso3(const char * src, char * dst, int64_t ne, cudaStream_t stream);
void ggml_cuda_cpy_f16_iso4(const char * src, char * dst, int64_t ne, cudaStream_t stream);
void ggml_cuda_cpy_f32_planar3(const char * src, char * dst, int64_t ne, cudaStream_t stream);
void ggml_cuda_cpy_f32_planar4(const char * src, char * dst, int64_t ne, cudaStream_t stream);
void ggml_cuda_cpy_f32_iso3(const char * src, char * dst, int64_t ne, cudaStream_t stream);
void ggml_cuda_cpy_f32_iso4(const char * src, char * dst, int64_t ne, cudaStream_t stream);
// Dequantize to F16 (for fattn-mma, matches to_fp16_cuda_t signature)
void dequantize_row_planar3_0_cuda(const void * x, half * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
void dequantize_row_planar4_0_cuda(const void * x, half * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
void dequantize_row_iso3_0_cuda(const void * x, half * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
void dequantize_row_iso4_0_cuda(const void * x, half * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

void ggml_cuda_cpy_f16_turbo3(const char * cx, char * cdst, const int ne, cudaStream_t stream);
void ggml_cuda_cpy_f32_turbo3(const char * cx, char * cdst, const int ne, cudaStream_t stream);
void ggml_cuda_cpy_f16_turbo4(const char * cx, char * cdst, const int ne, cudaStream_t stream);
void ggml_cuda_cpy_f32_turbo4(const char * cx, char * cdst, const int ne, cudaStream_t stream);
void ggml_cuda_cpy_f16_turbo2(const char * cx, char * cdst, const int ne, cudaStream_t stream);
void ggml_cuda_cpy_f32_turbo2(const char * cx, char * cdst, const int ne, cudaStream_t stream);
void ggml_cuda_cpy_f16_turbo3c(const char * cx, char * cdst, const int ne, cudaStream_t stream);
void ggml_cuda_cpy_f32_turbo3c(const char * cx, char * cdst, const int ne, cudaStream_t stream);

// Dequantise a contiguous turbo block array to F16.
// Signature matches to_fp16_cuda_t so these can be registered in
// ggml_get_to_fp16_cuda() for use by MMA/WMMA flash-attention kernels.
void dequantize_row_turbo3_0_cuda(const void * x, half * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
void dequantize_row_turbo4_0_cuda(const void * x, half * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
void dequantize_row_turbo2_0_cuda(const void * x, half * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
void dequantize_row_turbo3c_0_cuda(const void * x, half * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);

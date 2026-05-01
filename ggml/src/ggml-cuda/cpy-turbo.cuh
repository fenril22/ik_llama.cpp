#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

void ggml_cuda_cpy_f16_turbo3(const char * cx, char * cdst, const int ne, cudaStream_t stream);
void ggml_cuda_cpy_f32_turbo3(const char * cx, char * cdst, const int ne, cudaStream_t stream);
void ggml_cuda_cpy_f16_turbo4(const char * cx, char * cdst, const int ne, cudaStream_t stream);
void ggml_cuda_cpy_f32_turbo4(const char * cx, char * cdst, const int ne, cudaStream_t stream);
void ggml_cuda_cpy_f16_turbo2(const char * cx, char * cdst, const int ne, cudaStream_t stream);
void ggml_cuda_cpy_f32_turbo2(const char * cx, char * cdst, const int ne, cudaStream_t stream);

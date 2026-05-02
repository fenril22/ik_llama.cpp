// TurboQuant fattn-vec-f16 explicit instantiation

#include "../fattn-vec-f16.cuh"

DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0);
DECL_FATTN_VEC_F16_CASE(256, GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0);

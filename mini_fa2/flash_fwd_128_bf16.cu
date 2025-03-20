// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "flash_fwd_launch.h"

namespace flash {

template<>
void run_mini_mha_fwd_<cutlass::bfloat16_t, 128, false>(kernel_params &params, cudaStream_t stream) {
    run_mini_mha_fwd_128<cutlass::bfloat16_t, false>(params, stream);
}

} // namespace FLASH_NAMESPACE
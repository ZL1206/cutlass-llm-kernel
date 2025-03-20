// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "flash_fwd_launch.h"

namespace flash {

template<>
void run_mha_fwd_<cutlass::half_t, 128, true>(kernel_params &params, cudaStream_t stream) {
    run_mha_fwd_128<cutlass::half_t, true>(params, stream);
}

} // namespace FLASH_NAMESPACE
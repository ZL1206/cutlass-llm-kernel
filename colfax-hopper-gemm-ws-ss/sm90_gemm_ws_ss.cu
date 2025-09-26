#include "hopper_gemm_kernel_launch.h"

template void run_gemm_forward_<cutlass::half_t>(fwd_params &params, cudaStream_t stream);
  
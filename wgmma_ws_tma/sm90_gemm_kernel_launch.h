#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/cluster_launch.hpp"

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"





int run() {
    
}



void gemm_tn_launch(int m, int n, int k) {

    cudaDeviceProp props;
    int current_device_id;
    CUDA_CHECK(cudaGetDevice(&current_device_id));
    CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));

    if (props.major != 9 || props.minor != 0) {
        printf("This example requires a GPU of NVIDIA's Hopper Architecture (compute capability 90).\n");
        return 0;
    }


}
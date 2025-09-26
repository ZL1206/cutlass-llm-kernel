#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/cluster_launch.hpp"

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

#include "kernel_traits.h"
#include "tile_scheduler.hpp"

#include "flash.h"

#include "sm90_mainloop_tma_gmma_ws.hpp"
#include "sm90_epilogue_tma_ws.hpp"
#include "sm90_gemm_kernel.h"


template<typename T>
void run_gemm_forward_(fwd_params &params, cudaStream_t stream) {
    
    using Kernel_traits =  Kernel_traits<128, 256, 128, 2, 12>;

    using CollectiveMainloop = flash::CollectiveMainloop<Kernel_traits>;
    using CollectiveEpilogue = flash::CollectiveEpilogur<Kernel_traits>;

    using Scheduler = flash::SingleTileScheduler;

    typename CollectiveMainloop::Arguments mainloop_args {
        static_cast<T const*>(params.ptr_A),
        {params.M, params.K},
        {params.A_row_stride, _1{}},
        static_cast<T const*>(params.ptr_B),
        {params.N, params.K},
        {params.B_row_stride, _1{}}
    };

    typename CollectiveMainloop::Params mainloop_params = CollectiveMainloop::to_underlying_arguments(mainloop_args);

    typename CollectiveEpilogue::Arguments epilogue_args {
        static_cast<T*>(params.ptr_C),
        {params.M, params.N},
        {params.C_row_stride, _1{}}
    };

    typename CollectiveEpilogue::Params epilogue_params = CollectiveEpilogue::to_underlying_arguments(epilogue_args);

    int num_block_m = cutlass::ceil_div(params.M, Kernel_traits::kBlockM);
    int num_block_n = cutlass::ceil_div(params.N, Kernel_traits::kBlockN);

    num_block_m = cutlass::ceil_div(num_block_m, size<0>(ClusterShape{})) * size<0>(ClusterShape{});
    num_block_n = cutlass::ceil_div(num_block_n, size<1>(ClusterShape{})) * size<1>(ClusterShape{});

    typename Scheduler::Arguments scheduler_args = {num_block_m, num_block_n, 1};
    typename Scheduler::Params scheduler_params = Scheduler::to_underlying_arguments(scheduler_args);

    void* kernel = (void*)flash::hopper_gemm_ws<Kernel_traits, Scheduler>;

    int smem_size = Kernel_traits::kSmemSize;
    
    if (smem_size >= 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }

    int device;
    cudaGetDevice(&device);
    int multiprocessor_count;
    cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device);

    dim3 grid_dim = Scheduler::get_grid_dim(scheduler_args, multiprocessor_count);

    static constexpr int ctaSize = Kernel_traits::kNWarps * 32;

    dim3 block_dim(ctaSize);

    dim3 cluster_dim(size<0>(ClusterShape{}), size<1>(ClusterShape{}), size<2>(ClusterShape{}));
    
    cutlass::ClusterLaunchParams launch_params{grid_dim, block_dim, cluster_dim, smem_size, stream};

    cutlass::launch_kernel_on_cluster(launch_params, kernel, mainloop_params, epilogue_params, scheduler_params);


}



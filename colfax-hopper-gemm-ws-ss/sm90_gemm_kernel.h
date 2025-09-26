#pragma once

#include "cute/tensor.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "kernel_traits.h"
#include "tile_scheduler.hpp"
#include "sm90_mainloop_tma_gmma_ws.hpp"

namespace flash {

using namespace cute;

template<typename Kernel_traits, typename TileScheduler>
__global__ void
__launch_bounds__(Ktraits::kNWarps * cutlass::NumThreadsPerWarp, 1)
hopper_gemm_ws(CUTE_GRID_CONSTANT typename CollectiveMainloop<Ktraits>::Params const mainloop_params,
               CUTE_GRID_CONSTANT typename CollectiveEpilogue<Ktraits>::Params const epilogue_params,
               CUTE_GRID_CONSTANT typename TileScheduler::Params const scheduler_params) 
{

    using TileShape = typename Kernel_traits::TileShape;
    using ClusterShape = typename Kernel_traits::ClusterShape;
    
    static constexpr int NumMmaThreads = size(typename Kernel_traits::TiledMma{});

    static constexpr int NumCopyThreads = cutlass::NumThreadsPerWarpGroup;

    using CollectiveMainloop = CollectiveMainloop<Kernel_traits>;
    using CollectiveEpilogue = CollectiveEpilogue<Kernel_traits>;

    using MainloopPipeline = typename Kernel_traits::MainloopPipeline;
    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;

    extern __shared__ char smem_[];

    using SharedStorage = typename Kernel_traits::SharedStorage;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_);

    const int lane_predicate = cute::elect_one_sync();
    const int warp_idx = cutlass::canonical_warp_idx_sync();

    if (warp_idx == 0 && lane_predicate) {
        CollectiveMainloop::prefetch_tma_descriptors(mainloop_params);
    }

    const int warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;

    PipelineParams pipeline_params;
    pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytes;

    int warp_group_idx = cutlass::canonical_warp_group_idx();

    pipeline_params.role = warp_group_idx == 0 ？MainloopPipeline::ThreadCategory::Producer : MainloopPipeline::ThreadCategory::Consumer;
    pipeline_params.is_leader = warp_group_thread_idx == 0;
    pipeline_params.num_consumers = NumMmaThreads;
    
    if (warp_idx == 0 && lane_predicate) {
        shared_storage.barrier_C.init(size(ClusterShape{}) /*numThreads*/);
    }

    MainloopPipeline pipeline(shared_storage.pipeline, pipeline_params, ClusterShape{});

    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue;

    const int k_tile_count = cutlass::ceil_div(size<1>(mainloop_params.shape_A), Kernel_traits::kBlockK);

    if constexpr (size(ClusterShape{}) > 1) {
        cute::cluster_arrive_relaxed();
        cute::cluster_wait();
    } else {
        __syncthreads();
    }

    static_assert(Kernel_traits::kNWarps == 8 || Kernel_traits::kNWarps == 12);

    if (warp_group_idx == 0) { // producer
        cutlass::arch::warpgroup_reg_dealloc<Kernel_traits::kNWarps == 16 ? 32 : 24>();
        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        if (warp_idx_in_warpgroup == 0) { // load A,B in producer warp 0
            PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();

            int work_idx = 0;

            TileScheduler scheduler(&shared_storage.tile_count_semaphore);

            for (auto work_tile_info = scheduler.get_initial_work();
                 work_tile_info.is_valid(scheduler_params);
                 work_tile_infp = scheduler.template get_next_work</*IsProducer=*/true>(scheduler_params, work_tile_info)) {
                
                auto block_coord = work_tile_info.get_block_coord(scheduler_params);

                collective_mainloop.load(mainloop_params, pipeline, smem_pipe_write,
                                         shared_storage, scheduler, scheduler_params,
                                         work_tile_info, block_coord, work_idx, k_tile_count);
                ++work_idx;
            }
            collective_mainloop.load_tail(pipeline, smem_pipe_write);
        }
    } else { // consumer
        cutlass::arch::warpgroup_reg_alloc<Kernel_traits::kNWarps == 8 ? 256 : 240>();
        TileScheduler scheduler(&shared_storage.tile_count_semaphore);

        typename Kernel_traits::TiledMma tiled_mma;

        PipelineState smem_pipe_read;

        int work_idx = 0;
        for (auto work_tile_info = scheduler.get_initial_work();
             work_tile_info.is_valid(scheduler_params);
             work_tile_info = scheduler.template get_next_work</*IsProducer=*/false>(scheduler_params, work_tile_info)) {

            // gemm accumulator
            Tensor tCrC = partition_fragment_C(tiled_mma, select<0,1>(TileShape{}));
            clear(tCrC);

            auto block_coord = work_tile_info.get_block_coord(scheduler_params);

            collective_mainloop.mma(mainloop_params, pipeline, smem_pipe_read,
                                    tCrC, threadIdx.x - NumCopyThreads, work_id,
                                    shared_storage, k_tile_count);
            collective_epilogue.store(epilogue_params, tCrC, shared_storage, tiled_mma,
                                      threadIdx.x - NumCopyThreads, block_coord);
            
            ++work_idx;
        }

        collective_epilogue.store_tail();
    }


}



} // namespace flash


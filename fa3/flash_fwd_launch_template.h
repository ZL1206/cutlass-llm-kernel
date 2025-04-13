#pragma once

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"  // For device_kernel
#include <cutlass/kernel_hardware_info.h>
#include "cutlass/cluster_launch.hpp"
#include "cutlass/kernel_launch.h"

#include "static_switch.h"
#include "flash.h"
#include "tile_size.h"
#include "tile_scheduler.hpp"
#include "flash_fwd_kernel_sm90.h"
#include "flash_fwd_kernel_sm80.h"
#include "mainloop_fwd_sm90_tma_gmma_ws.hpp"
#include "mainloop_fwd_sm80.hpp"
#include "epilogue_fwd.hpp"



using namespace cute;

template<typename T, int kHeadDim, bool Is_causal, bool Varlen, bool Split>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {

    static constexpr std::tuple<int, int, int, int, bool> kBlockMN_kNWarps_Stages_RS = tile_size_fwd_sm8x(Arch == 86 || Arch == 89, kHeadDim, Is_causal, sizeof(T), Varlen && Split);
    static constexpr int kBlockM = std::get<0>(kBlockMN_kNWarps_Stages_RS);
    static constexpr int kBlockN = std::get<1>(kBlockMN_kNWarps_Stages_RS);

    static constexpr int kNWarps = std::get<2>(kBlockMN_kNWarps_Stages_RS);
    static constexpr int kStages = Arch >= 90 ? 2 : std::get<3>(kBlockMN_kNWarps_Stages_RS);
    static constexpr bool Q_in_regs = Arch >= 90 ? false : std::get<4>(kBlockMN_kNWarps_Stages_RS);

    using TileShape_MNK = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;

    using CollectiveMainloop = flash::CollectiveMainloopFwdSm80<kNWarps, kStages, Q_in_regs, TileShape_MNK, kHeadDimV, Element, float, cutlass::arch::Sm80, Is_causal, Is_local, Has_softcap, Varlen, PagedKVNonTMA, AppendKV, PackGQA, Split>;

    static constexpr int NumProducerThreads = CollectiveMainloop::NumMmaThreads;

    using SchedulerPersistent = std::conditional_t<Varlen,
        flash::VarlenDynamicPersistentTileScheduler<kBlockM, CollectiveMainloop::NumMmaThreads, NumProducerThreads, Split, PackGQA, Arch >= 90 /*WarpSpecialized*/>,
        std::conditional_t<!Is_causal && !Is_local,
            flash::StaticPersistentTileScheduler<Split>,
            flash::DynamicPersistentTileScheduler<CollectiveMainloop::NumMmaThreads, NumProducerThreads, Split, PackGQA, Arch >= 90 /*WarpSpecialized*/>
        >
    >;

    static constexpr bool UsePersistentScheduler = ((Is_causal && !Varlen) || (Varlen && Split));

    using Scheduler = std::conditional_t<!UsePersistentScheduler, SchedulerSingleTile, SchedulerPersistent>;

    using AttnKernel = flash::enable_sm80_to_sm89<flash::FlashAttnFwdSm80<CollectiveMainloop, CollectiveEpilogue, Scheduler>>;

    bool const is_varlen_q = params.cu_seqlens_q;
    bool const is_varlen_k = params.cu_seqlens_k;
    int seqlen_q = !is_varlen_q ? params.seqlen_q : params.total_q;

    int batch_q = !is_varlen_q ? params.b : 1;
    int batch_k = !is_varlen_k ? (params.kv_batch_idx ? params.b_k : params.b) : 1;

    int qhead_per_khead = cutlass::ceil_div(params.h, params.h_k);

    int num_blocks_m = cutlass::ceil_div(params.seqlen_q * qhead_per_khead, get<0>(TileShape_MNK{}));

    if (Varlen && params.num_splits_dynamic_ptr && !params.skip_scheduler_metadata_computation) {
        prepare_varlen_num_blocks(params, stream, PackGQA, kBlockM, kBlockN, Arch >= 90 /*enable_pdl*/);
        CHECK_CUDA_KERNEL_LAUNCH();
    }


    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    typename AttnKernel::Params kernel_params = AttnKernel::to_underlying_arguments({
        mainloop_args, epilogue_args, {device, params.num_sm}, scheduler_args
    });

    dim3 grid_dims = AttnKernel::get_grid_shape(kernel_params);
    dim3 block_dims = AttnKernel::get_block_shape();
    int smem_size = AttnKernel::SharedStorageSize;

    auto kernel = cutlass::device_kernel<AttnKernel>;
    if (smem_size >= 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    // kernel<<<grid_dims, block_dims, smem_size, stream>>>(kernel_params);
    cutlass::kernel_launch<AttnKernel>(grid_dims, block_dims, smem_size, stream, kernel_params,
                                       Arch >= 90 && Varlen && params.num_splits_dynamic_ptr && !params.skip_scheduler_metadata_computation /*launch_with_pdl*/);


}









template<typename T, int kHeadDim, bool Is_causal, bool Split>
void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream) {
   
    VARLEN_SWITCH(params.cu_seqlens_q || params.cu_seqlens_k, Varlen, [&] {
                // Only needed here to decide if we should use cluster
        static constexpr int kBlockM = 16;         
        run_flash_fwd<kHeadDim, T, Is_causal, Varlen, Split>(params, stream);         
    });
}

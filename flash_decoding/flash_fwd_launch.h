#pragma once

#include "hardware_info.h"
#include "static_switch.h"
#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include "int4_kernel_traits.h"
#include "utils.h"
#include "softmax.h"
#include "mask.h"
#include "flash.h"
#include <c10/cuda/CUDAException.h>

namespace flash {

template<typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool Split>
__global__ void flash_fwd_splitkv_kernel(__grid_constant__ const kernel_params params) {
    const int m_block = blockIdx.x;
    const int bidb = Split ? blockIdx.z / params.h : blockIdx.y;
    const int bidh = Split ? blockIdx.z - bidb * params.h : blockIdx.z;
    const int n_split_idx = Split ? blockIdx.y : 0;
    const int num_n_splits = Split ? gridDim.y : 1;
    using T = typename Kernel_traits::T;
    using Tkv = typename Kernel_traits::Tkv;

    const int tid = threadIdx.x;
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);

    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = 0;
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if (Is_causal) {
        n_block_max = cute::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q, kBlockN));
    }

    const int* block_table = params.block_table + bidb * params.block_table_batch_stride;
    const int block_table_idx = (n_block_max - 1) * kBlockN;
    const int64_t row_offset_k = block_table[block_table_idx] * params.k_block_stride + (bidh / params.gqa) * params.k_head_stride;
    const int64_t k_params_offset = block_table[block_table_idx] * params.k_params_block_stride + (bidh / params.gqa) * params.k_params_head_stride;
    
    Tensor mQ = make_tensor(make_gemm_ptr(reinterpret_cast<T*>(params.q_ptr) + binfo.q_offset(params.q_row_stride)), 
                                          make_shape(binfo.actual_seqlen_q, params.h, params.d),
                                          make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));
    
    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.begin()),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.begin()),
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.begin()), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});

    Tensor gKP = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(params.k_params_ptr) + k_params_offset),
                             Shape<_2, Int<kBlockN>>{},
                             Stride<Int<kBlockN>, _1>{});
    
    Tensor gVP = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(params.v_params_ptr) + k_params_offset),
                             Shape<_2, Int<kBlockN>>{},
                             Stride<Int<kBlockN>, _1>{});
    
    Tensor sKP = make_tensor(make_smem_ptr(shared_storage.smem_k_params.begin()),
                            typename Kernel_traits::SmemLayoutKVParams{});
    
    Tensor sVP = make_tensor(make_smem_ptr(shared_storage.smem_v_params.begin()),
                            typename Kernel_traits::SmemLayoutKVParams{});


    typename Kernel_traits::GmemTiledCopyQ gmem_tiled_copy_Q;
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_slice(idx);
    Tensor tQgQ = gmem_thr_copy_Q.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ);

    typename Kernel_traits::GmemTiledCopyKV gmem_tiled_copy_KV;
    auto gmem_thr_copy_KV = gmem_tiled_copy_KV.get_slice(idx);
    Tensor tKgK = gmem_thr_copy_KV.partition_S(gK);  
    Tensor tKsK = gmem_thr_copy_KV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_KV.partition_S(gV);  
    Tensor tVsV = gmem_thr_copy_KV.partition_D(sV);

    Tensor tKPgKP = gmem_thr_copy_Q.partition_S(gKP);  
    Tensor tKPsKP = gmem_thr_copy_Q.partition_D(sKP);
    Tensor tVPgVP = gmem_thr_copy_Q.partition_S(gVP);  
    Tensor tVPsVP = gmem_thr_copy_Q.partition_D(sVP);

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_KV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    Tensor cKVP = make_identity_tensor(make_shape(size<0>(sKP), size<1>(sKP)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor tKVPcKVp = gmem_thr_copy_Q.partition_S(cKVP);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(idx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK_q  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tSrK = make_tensor<T>(Shape< Shape<_2, _2>, _2, _8>{},              // (MMA,MMA_N,MMA_K)
                                 Stride< Stride<_1, _2>, _32, _4>{});
    Tensor tSrK_dq = make_tensor(tSrK.data(), Layout<Shape<_8, _2, Shape<_2, _2>>, Stride<_1, _32, Stride<_8, _16>>>{});
    Tensor k_params = make_tensor<T>(Shape<_2, _2>{},
                                    Stride<_1, _2>{});
    
    typename Kernel_traits::TiledMma_PV tiled_mma_pv;
    auto thr_mma_pv = tiled_mma_pv.get_thread_slice(idx);
    Tensor tOrVt_q  = thr_mma_pv.partition_fragment_B(sVt);
    Tensor tOrVt = make_tensor<T>(Shape< Shape<_2, _2>, _16, _1>{}, 
                                  Stride< Stride<_1, _32>, _2, _0>{});
    Tensor tOrVt_dq = make_tensor(tOrVt.data(), Layout<Shape<Shape<_8, _2>, _4, _1>, Stride<Stride<_1, _32>, _8, _0>>{});
    Tensor v_params = make_tensor<T>(Shape<_2, Shape<_2, _2>, _1>{},
                                    Stride<_1, Stride<_2, _4>, _8>{});



    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(idx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(idx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_pv);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(idx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

    int n_block = n_block_max - 1;

    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tQgQ, tQsQ, tQcQ, tQpQ, params.seqlen_q);










}





template<typename Kernel_traits, bool Is_causal>
void run_flash_splitkv_fwd(kernel_params &params, cudaStream_t stream) {
    
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        BOOL_SWITCH(is_even_K, IsEvenKConst, [&] {
            BOOL_SWITCH(params.num_splits > 1, Split, [&] {  
                // If Append_KV, then we must have seqlen_offsets, which means cu_seqlens_k != nullptr.
                // If not IsEvenKConst, we also set IsEvenMNConst to false to reduce number of templates.
                                
                auto kernel = &flash_fwd_splitkv_kernel<Kernel_traits, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && !Append_KV && IsEvenKConst && !Is_local && Kernel_traits::kHeadDim <= 128, IsEvenKConst, Is_softcap, Split, Append_KV>;
                // auto kernel = &flash_fwd_splitkv_kernel<Kernel_traits, Is_causal, false, true, Split, Append_KV>;
                // auto kernel = &flash_fwd_splitkv_kernel<Kernel_traits, Is_causal, false, IsEvenKConst>;
                if (smem_size >= 48 * 1024) {
                    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                }
                kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
                                
            });
            
        });
    });
    
}






template<typename T, int Headdim, bool Is_causal>
void run_mha_fwd_splitkv_dispatch(kernel_params &params, cudaStream_t stream) {
    constexpr static int kBlockN = 64;
    if (params.max_seqlen_q < 16) {
        run_flash_splitkv_fwd<Kernel_traits<T, 16, kBlockN, Headdim, 4>, Is_causal>(params, stream);
    } else {
        run_flash_splitkv_fwd<Kernel_traits<T, 32, kBlockN, Headdim, 4>, Is_causal>(params, stream);
    }
    
}


} // namespace
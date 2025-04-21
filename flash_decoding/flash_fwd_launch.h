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
#include "block_info.h"
#include <c10/cuda/CUDAException.h>

namespace flash {

using namespace cute;

template<typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool Split>
__global__ void flash_fwd_splitkv_kernel(__grid_constant__ const kernel_params params) {
    using T = typename Kernel_traits::T;
    using Tkv = typename Kernel_traits::Tkv;
    using SharedStorage = typename Kernel_traits::TensorStorage;
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;


    extern __shared__ char smem[];
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem);

    const int m_block = blockIdx.x;
    const int bidb = Split ? blockIdx.z / params.h : blockIdx.y;
    const int bidh = Split ? blockIdx.z - bidb * params.h : blockIdx.z;
    const int n_split_idx = Split ? blockIdx.y : 0;
    const int num_n_splits = Split ? gridDim.y : 1;
    
    const int idx = threadIdx.x;
    const int warp_idx = idx / 32;
    const int lane = idx % 32;
    
    if (thread0()) {
        printf("m_block is %d, bidb is %d, bidh is %d, n_split_idx is %d, num_n_splits is %d\n", m_block, bidb, bidh, n_split_idx, num_n_splits);
    }
    __syncthreads();
    const BlockInfo binfo(params, bidb);

    
    
    
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = 0;
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if (Is_causal) {
        n_block_max = cute::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q, kBlockN));
    }

    if (thread0()) {
        printf("sum_s_q is %d, actual_seqlen_q is %d, actual_seqlen_k is %d, n_block_max is %d\n", binfo.sum_s_q, binfo.actual_seqlen_q, binfo.actual_seqlen_k, n_block_max);
    }
    __syncthreads();

    
    const int* block_table = params.block_table + bidb * params.block_table_batch_stride;
    const int block_table_idx = n_block_max - 1;
    const int64_t row_offset_k = block_table[block_table_idx] * params.k_block_stride + (bidh / params.gqa) * params.k_head_stride;
    const int64_t k_params_offset = block_table[block_table_idx] * params.k_params_block_stride + (bidh / params.gqa) * params.k_params_head_stride;
    
    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(params.q_ptr) + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)), 
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(params.k_ptr) + row_offset_k)),
                            Shape<Int<kBlockN>, Int<32>>{},
                            make_stride(Int<32>{}, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(params.v_ptr) + row_offset_k)),
                            Shape<Int<kBlockN>, Int<32>>{},
                            make_stride(Int<32>{}, _1{}));
    
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
    // 应该删掉这个
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

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

    Tensor acc_o = partition_fragment_C(tiled_mma_pv, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K
    clear(acc_o);

    int n_block = n_block_max - 1;

    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tQgQ, tQsQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);

    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_KV, tKgK, tKsK, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN);
    
    flash::copy</*Is_even_MN=*/false>(gmem_tiled_copy_Q, tKPgKP, tKPsKP, tKVPcKVp, 2);

    cute::cp_async_fence();
    
    flash::Softmax<2 * size<1>(acc_o)> softmax;
    flash::Mask mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q);

    int n_masking_steps = 0;
    if (!Is_causal && !Is_even_MN) {
        n_masking_steps = 1;
    } else if (Is_causal) {
        n_masking_steps = n_block_max - cute::max((m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q) / kBlockN, 0);
    }

    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        // async load v
        if (n_block == n_block_max - 1) {
            flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(gmem_tiled_copy_KV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN);
            flash::copy</*Is_even_MN=*/false>(gmem_tiled_copy_Q, tVPgVP, tVPsVP, tKVPcKVp, 2);
        } else {
            tVgV.data() = tVgV.data() + (block_table[n_block] - block_table[n_block + 1]) * params.v_block_stride;
            tVPgVP.data() = tVPgVP.data() + (block_table[n_block] - block_table[n_block + 1]) * params.v_params_block_stride;
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_KV, tVgV, tVsV, tKVcKV, tKVpKV);
        }
        cute::cp_async_fence();

        if (thread0()) {
            print("sK is:\n");
            print_tensor(sK);
            print("sKP is:\n");
            print_tensor(sKP);
        }

        // compute qk
        flash::gemm<T, Tkv>(acc_s, tSrQ, tSrK_q, tSrK, tSrK_dq, tSsQ, tSsK, k_params, sKP, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

        
        if (n_masking_steps > 0) {
            mask.template apply_mask<Is_causal, Is_even_MN>(acc_s, n_block * kBlockN, m_block * kBlockM, idx, kNWarps * 16);
            softmax.template softmax_rescale_o</*Check_inf=*/Is_causal>(acc_s, acc_o, params.scale_softmax_log2);
            n_masking_steps = n_masking_steps - 1;
        } else {
            mask.template apply_mask</*Causal_mask=*/false>(acc_s, n_block * kBlockN, m_block * kBlockM, idx, kNWarps * 16);
            softmax.template softmax_rescale_o(acc_s, acc_o, params.scale_softmax_log2);
        }
        
        // wait v
        flash::cp_async_wait<0>();
        __syncthreads();

        // async load next k
        if (n_block > n_block_min) {
            tKgK.data() = tKgK.data() + (block_table[n_block - 1] - block_table[n_block]) * params.k_block_stride;
            tKPgKP.data() = tKPgKP.data() + (block_table[n_block - 1] - block_table[n_block]) * params.k_params_block_stride;
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_KV, tKgK, tKsK, tKVcKV, tKVpKV);
            cute::cp_async_fence();
        }
        cute::cp_async_fence();

    
        Tensor rP = make_tensor_like<T>(acc_s);
        convert_type_out(acc_s, rP);

        // second gemm, change acc_s layout, output as input
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));

        flash::gemm_rs<T, Tkv>(acc_o, tOrP, tOrVt_q, tOrVt, tOrVt_dq, tOsVt, v_params, sVP, tiled_mma_pv, smem_tiled_copy_V, smem_thr_copy_V);
    }

    // Epilogue

    // warp lse
    Tensor lse = softmax.template normalize_softmax_lse(acc_o, params.scale_softmax);
    if (thread0()) {
        print("lse: \n");
        print_tensor(lse);
    }
    // block lse
    Tensor smem_lse = make_tensor(make_smem_ptr(reinterpret_cast<float*>(smem)), typename Kernel_traits::SmemLayoutLse{});
    Tensor final_lse = softmax.template normalize_final_lse(lse, smem_lse, acc_o, idx);

    // convert acc_o to fp16
    Tensor rO = make_tensor_like<T>(acc_o);
    flash::convert_type_out(acc_o, rO);
    

    // copy acc_o to shared memory
    Tensor mO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
    Tensor sO = local_tile(mO(_, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                        make_coord(warp_idx, 0));  // (kBlockM, kHeadDim)

    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma_pv);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(lane);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    __syncthreads();


    // smem to global
    const int64_t o_offset = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(params.o_ptr) + o_offset),
                    Shape<Int<kBlockM>, Int<kHeadDim>>{},
                    make_stride(params.o_head_stride, Int<1>{}));
    
    const int64_t lse_offset = bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb) + m_block * kBlockM;
    Tensor gLse = make_tensor(make_gmem_ptr(reinterpret_cast<float*>(params.softmax_lse_ptr) + lse_offset),
                              Shape<Int<kBlockM>>{}, 
                              Stride<_1>{});
    
    // reduction
    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(idx);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
    Tensor tOsO = gmem_thr_copy_O.partition_S(mO);

    Tensor tOrO_accum = make_tensor<T>(shape(tOgO));
    clear(tOrO_accum);
    
    for (int i = 0; i < 4; i++) {
        Tensor tOrO = make_tensor<T>(shape(tOgO));
        const int row = size<1>(tOrO);

        for (int mi = 0; mi < row; mi++) {
            cute::copy(gmem_tiled_copy_O, tOsO(_, i * row + mi, _), tOrO(_, mi, _));
        }
        for (int j = 0; j < size(tOrO); j++) {
            tOrO_accum(j) += tOrO(j);
        }
    }


    // save lse
    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma_pv.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    static_assert(decltype(size<0>(taccOcO))::value == 4);
    // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
    Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
    CUTE_STATIC_ASSERT_V(size(final_lse) == size(taccOcO_row));                     // MMA_M
    if (warp_idx == 0 && get<1>(taccOcO_row(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(final_lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLse(row) = final_lse(mi); }
        }
    }

    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));

    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO_accum, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM);
    
    //#endif
}





template<typename Kernel_traits, bool Is_causal>
void run_flash_splitkv_fwd(kernel_params &params, cudaStream_t stream) {
    
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    printf("smem_size is %d, num_m_block is %d, params.b is %d, params.h is %d, gqa is %d, is_even_MN is %d\n", smem_size, num_m_block,  params.b, params.h, params.gqa, is_even_MN);
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        BOOL_SWITCH(is_even_K, IsEvenKConst, [&] {
            BOOL_SWITCH(params.num_splits > 1, Split, [&] {  
                // If Append_KV, then we must have seqlen_offsets, which means cu_seqlens_k != nullptr.
                // If not IsEvenKConst, we also set IsEvenMNConst to false to reduce number of templates.
                                
                auto kernel = &flash_fwd_splitkv_kernel<Kernel_traits, Is_causal, IsEvenMNConst, IsEvenKConst, Split>;
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
    printf("params.seqlen_q is %d\n", params.seqlen_q);
    if (params.seqlen_q <= 16) {
        run_flash_splitkv_fwd<Kernel_traits<T, cutlass::uint4b_t, 16, kBlockN, Headdim, 4>, Is_causal>(params, stream);
    } else {
        run_flash_splitkv_fwd<Kernel_traits<T, cutlass::uint4b_t, 32, kBlockN, Headdim, 4>, Is_causal>(params, stream);
    }
    
}


} // namespace
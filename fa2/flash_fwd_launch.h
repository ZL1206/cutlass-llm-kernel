#pragma once

#include "hardware_info.h"
#include "static_switch.h"
#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include "kernel_traits.h"
#include "utils.h"
#include "softmax.h"
#include "mask.h"
#include "flash.h"
#include "blockinfo.h"
#include <c10/cuda/CUDAException.h>

namespace flash {

using namespace cute;

template<typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool IsVarlen>
__global__ void flash_fwd_kernel(__grid_constant__ const kernel_params params) {

    using T = typename Kernel_traits::T;
    const int m_block = blockIdx.x;
    const int batch = blockIdx.y;
    const int head = blockIdx.z;
    const int idx = threadIdx.x;
    if (thread0()) {
        printf("m_block is %d, batch is %d, head is %d, idx is %d\n", m_block, batch, head, idx);
    }

    extern __shared__ char smem[];
    
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    
    const BlockInfo</*Varlen=*/IsVarlen> binfo(params, batch);
    
    if (m_block * kBlockM > binfo.actual_seqlen_q) return;

    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    const int n_block_min = 0;
    if (Is_causal) {
        n_block_max = cute::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q, kBlockN));
    }
    
    
    if ((Is_causal || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(params.o_ptr) + binfo.q_offset(params.o_batch_stride, params.o_row_stride, batch)),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, head, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                              make_coord(m_block, 0));  // (kBlockM, kHeadDim)
        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(idx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<T>(shape(tOgO));
        clear(tOrO);
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        return;
    }
    

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(params.q_ptr) + binfo.q_offset(params.q_batch_stride, params.q_row_stride, batch)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, head, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                        make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    
    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(params.k_ptr) + binfo.k_offset(params.k_batch_stride, params.k_row_stride, batch)),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.k_row_stride, params.k_head_stride, _1{}));
    
    Tensor gK = local_tile(mK(_, head / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                        make_coord(_, 0));  // (kBlockN, kHeadDim, n)
    
    Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(params.v_ptr) + binfo.q_offset(params.v_batch_stride, params.v_row_stride, batch)),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.v_row_stride, params.v_head_stride, _1{}));
    
    Tensor gV = local_tile(mV(_, head / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                        make_coord(_, 0));  // (kBlockN, kHeadDim, n)
    
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem)),
                        typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data() + size(sQ),
                        typename Kernel_traits::SmemLayoutKV{});
    
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});

    // global to shared memory
    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_slice(idx);
    
    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
    // prediction
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // register used in mma
    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(idx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVt);                // (MMA, MMA_K,MMA_N)

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

    // shared memory to register
    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(idx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(idx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(idx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    // async copy q
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                       binfo.actual_seqlen_q - m_block * kBlockM);
    int n_block = n_block_max - 1;
    // async load k
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV,
                                       binfo.actual_seqlen_k - n_block * kBlockN);
    cute::cp_async_fence();

    clear(acc_o);

    flash::Softmax<2 * size<1>(acc_o)> softmax;
    flash::Mask mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q);

    // 可以精确的计算出需要mask的步数
    int n_masking_steps = 0;
    if (!Is_causal && !Is_even_MN) {
        n_masking_steps = 1;
    } else if (Is_causal) {
        n_masking_steps = n_block_max - cute::max((m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q) / kBlockN, 0);
    }
    if (thread0()) {
        printf("n_block is %d, n_masking_steps is %d\n", n_block, n_masking_steps);
    }
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block == n_block_max - 1) {
            flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);
        }
        cute::cp_async_fence();

        // compute qk
        flash::gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, 
                    smem_tiled_copy_Q, smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);
        
        // wait v
        flash::cp_async_wait<0>();
        __syncthreads();
        // async load next k
        if (n_block > n_block_min) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
            cute::cp_async_fence();
        }

        if (n_masking_steps > 0) {
            mask.template apply_mask<Is_causal, Is_even_MN>(acc_s, n_block * kBlockN, m_block * kBlockM, idx, kNWarps * 16);
            softmax.template softmax_rescale_o</*Check_inf=*/Is_causal>(acc_s, acc_o, params.scale_softmax_log2);
            n_masking_steps = n_masking_steps - 1;
        } else {
            mask.template apply_mask</*Causal_mask=*/false>(acc_s, n_block * kBlockN, m_block * kBlockM, idx, kNWarps * 16);
            softmax.template softmax_rescale_o(acc_s, acc_o, params.scale_softmax_log2);
        }

        // Convert acc_s from fp32 to fp16/bf16
        //Tensor rP = flash::convert_type<T>(acc_s);
        
        constexpr int numel = decltype(size(acc_s))::value;
        cutlass::NumericArrayConverter<T, float, numel> convert_op;
        auto frag = convert_op(*reinterpret_cast<const cutlass::Array<float, numel> *>(acc_s.data()));
        Tensor rP = make_tensor(make_rmem_ptr<T>(&frag), acc_s.layout());
        
        if (thread0()) {
            printf("rP: "); print(rP); print("\n");
            print_tensor(rP);
        }
        if (thread0()) {
            print_tensor(sV);
        }
        // compute pv
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }

    // Epilogue
    if (thread0()) {
        printf("acc_o: "); print(acc_o); print("\n");
        print_tensor(acc_o);
    }
    Tensor lse = softmax.template normalize_softmax_lse(acc_o, params.scale_softmax);
    if (thread0()) {
        printf("acc_o: "); print(acc_o); print("\n");
        print_tensor(acc_o);
    }
    // Convert acc_o from fp32 to fp16/bf16
    //Tensor rO = flash::convert_type<T>(acc_o);
    constexpr int numel_ = decltype(size(acc_o))::value;
    cutlass::NumericArrayConverter<T, float, numel_> convert_op_;
    auto frag_ = convert_op_(*reinterpret_cast<const cutlass::Array<float, numel_> *>(acc_o.data()));
    Tensor rO = make_tensor(make_rmem_ptr<T>(&frag_), acc_o.layout());
    if (thread0()) {
        printf("rO: "); print(rO); print("\n");
        print_tensor(rO);
    }
    // copy register to shared memory
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(idx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    
    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(params.o_ptr) + binfo.q_offset(params.o_batch_stride, params.o_row_stride, batch)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, head, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(idx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    // shared memory to register
    Tensor tOrO = make_tensor<T>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));

    #pragma unroll
    for (int k = 0; k < size(tOpO); ++k) {
        tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
    }

    // register to global memory
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM);
    
}





template<typename Kernel_traits, bool Is_causal>
void run_flash_fwd(kernel_params &params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    // printf("smem_size = %d\n", smem_size);

    // Work-around for gcc 7. It doesn't like nested BOOL_SWITCH.
    // https://github.com/kokkos/kokkos-kernels/issues/349
    // https://github.com/HazyResearch/flash-attention/issues/21
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    const bool is_even_MN = params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool is_varlen = params.cu_seqlens_q != nullptr;
    printf("num_m_block is %d, batch is %d, head is %d, is_varlen %d\n", num_m_block, params.b, params.h, is_varlen);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,
        cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        BOOL_SWITCH(is_even_K, IsEvenKConst, [&] {
            BOOL_SWITCH(is_varlen, IsVarlen, [&] {
                // If not IsEvenKConst, we also set IsEvenMNConst to false to reduce number of templates.
                // If head dim > 128, set IsEvenMNConst to false to reduce number of templates
                auto kernel = &flash_fwd_kernel<Kernel_traits, Is_causal, IsEvenMNConst && IsEvenKConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst, IsVarlen>;
                if (smem_size >= 48 * 1024) {
                    C10_CUDA_CHECK(cudaFuncSetAttribute(
                        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                }
                kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                cudaDeviceSynchronize();
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,
                    cudaGetErrorString(err));
                    exit(EXIT_FAILURE);
                }
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
        });
    });
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,
             cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


template<typename T, bool Is_causal>
void run_mha_fwd_128(kernel_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x = cc_major == 8 && cc_minor > 0;
        
    // For sm86 or sm89, 64 x 64 is the fastest for causal (because it's square),
    // and 128 x 32 (48 KB smem) is the fastest for non-causal since we get 2 CTAs per SM.
    if (is_sm8x) {
        if constexpr(!Is_causal) {
            run_flash_fwd<Kernel_traits<T, 128, 32, Headdim, 4>, Is_causal>(params, stream);
        } else {
            run_flash_fwd<Kernel_traits<T, 64, 64, Headdim, 4>, Is_causal>(params, stream);
        }
    } else {
        run_flash_fwd<Kernel_traits<T, 128, 64, Headdim, 4>, Is_causal>(params, stream);
    }
         
}


}

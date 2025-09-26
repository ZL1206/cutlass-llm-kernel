#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/pipeline/pipeline.hpp"
#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"

namespace flash {


using namespace cute;

template<typename Kernel_traits>
struct  CollectiveMainloop
{
    using Element = typename Kernel_traits::Element;
    using TileShape = typename Kernel_traits::TileShape;

    static constexpr int kStages = Kernel_traits::kStages;

    using SmemLayoutAtomA = typename Kernel_traits::SmemLayoutAtomA;
    using SmemLayoutAtomB = typename Kernel_traits::SmemLayoutAtomB;
    using GmemTiledCopyA = typename Kernel_traits::GmemTiledCopyA;
    using GmemTiledCopyB = typename Kernel_traits::GmemTiledCopyB;
    using ClusterShape = typename Kernel_traits::ClusterShape;
    using KernelSchedule = typename Kernel_traits::KernelSchedule;
    using DispatchPolicy = MainloopSm90TmaGmmaWarpSpecialized<kStages, ClusterShape, KernelSchedule>;

    using TiledMma = typename Kernel_traits::TiledMma;

    using MainloopPipeline = cutlass::PipelineTmaAsync<kStages>;

    static_assert(cute::rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
    static_assert(size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{}) == 0, "SmemLayoutAtom must evenly divide tile shape.");
    static_assert(size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{}) == 0, "SmemLayoutAtom must evenly divide tile shape.");

    static_assert(cute::rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
    static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
    static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

    using SmemLayoutA = decltype(tile_to_shape(
                            SmemLayoutAtomA{},
                            make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<kStages>{}),
                            Step<_1, _2, _3>{}));
    using SmemLayoutB = decltype(tile_to_shape(
                            SmemLayoutAtomB{},
                            make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<kStages>{}),
                            Step<_1, _2, _3>{}));
    
    static_assert(kStages >= 2, "Specialization requires Stages set to value 2 or more.");

    static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>, "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
    static_assert(cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>, "GmemTiledCopy - invalid SM90 TMA copy atom specified.");

    struct SharedStorage {
        struct TensorStorage : cute::aligned_struct<128, _0> {
            cute::array_aligned<typename TiledMma::ValTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
            cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
        } tensors;
        using PipelineStorage = typename MainloopPipeline::SharedStorage;
        PipelineStorage pipeline;
    }

    using TensorStorage = typename SharedStorage::TensorStorage;
    using PipelineStorage = typename SharedStorage::PipelineStorage;

    // Host side kernel arguments
    struct Arguments {
        Element const* ptr_A;
        StrideA dA;
        Element const* ptr_B;
        StrideB dB;
        uint32_t mma_promotion_interval = 4;
    };

    // Device side kernel params
    struct Params {
        using TMA_A = decltype(make_tma_copy_A_sm90(
                        GmemTiledCopyA{},
                        make_tensor(static_cast<InternalElementA const*>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}), 
                        SmemLayoutA{}(_ , _, Int<0>{}),
                        TileShape{},
                        ClusterShape{}
        ));
        using TMA_B = decltype(make_tma_copy_B_sm90(
                        GmemTiledCopyB{},
                        make_tensor(static_cast<InternalElementB const*>(nullptr), repeat_like(StrideB{}, int32_t(0)), StrideB{}),
                        SmemLayoutB{}(_, _, Int<0>{}),
                        TileShape{},
                        ClusterShape{}
        ));

        TMA_A tma_load_a;
        TMA_B tma_load_b;
        uint32_t tma_transaction_bytes = TmaTransactionBytes;
        uint32_t tma_transaction_bytes_mk = TmaTransactionBytesMK;
        uint32_t tma_transaction_bytes_nk = TmaTransactionBytesNK;
    };

    //
    // Methods
    //
    template <class ProblemShape>
    static Params
    to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args) {
        auto [M, N, K] = problem_shape;
        
        Tensor tensor_a = make_tensor(make_gmem_ptr(args.ptr_A), make_shape(M, K), args.dA);
        Tensor tensor_b = make_tensor(make_gmem_ptr(args.ptr_B), make_shape(N, K), args.dB);

        typename Params::TMA_A tma_load_a = make_tma_copy_A_sm90(
            GmemTiledCopyA{},
            tensor_a,
            SmemLayoutA{}(_, _, Int<0>{}),
            TileShape{},
            ClusterShape{}
        );
        typename Params::TMA_B tma_load_b = make_tma_copy_B_sm90(
            GmemTiledCopyB{},
            tensor_b,
            SmemLayoutB{}(_, _, Int<0>{}),
            TileShape{},
            ClusterShape{}
        );
        uint32_t transaction_bytes_mk = TmaTransactionBytesMK;
        uint32_t transaction_bytes_nk = TmaTransactionBytesNK;
        uint32_t transaction_bytes = transaction_bytes_mk + transaction_bytes_nk;

        return {
            tma_load_a,
            tma_load_b,
            transaction_bytes,
            transaction_bytes_mk,
            transaction_bytes_nk
        };
    }

    static constexpr int K_PIPE_MAX = kStages;
    static constexpr int K_PIPE_MMAS = 1;
    static constexpr uint32_t TmaTransactionBytesMK = 
        cutlass::bits_to_bytes(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) * static_cast<uint32_t>(sizeof_bits<Element>::value));
    static constexpr uint32_t TmaTransactionBytesNK = 
        cutlass::bits_to_bytes(size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) * static_cast<uint32_t>(sizeof_bits<Element>::value));

    static constexpr uint32_t TmaTransactionBytes = TmaTransactionBytesMK + TmaTransactionBytesNK;

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& mainloop_params) {
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_a.get_tma_descriptor());
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_b.get_tma_descriptor());
    }

    template <class ProblemShape>
    CUTLASS_DEVICE auto
    load_init(ProblemShape const& problem_shape, Params const& mainloop_params) const {
        auto [M, N, K] = problem_shape;
        
        Tensor mA_mk = mainloop_params.tma_load_a,get_tma_tensor(make_shape(M, K));
        Tensor mB_nk = mainloop_params.tma_load_b,get_tma_tensor(make_shape(N, K));

        Tensor gA_mk = local_tile(mA_mk, TileShape{}, make_coord(_, _, _), Step<_1, X, _1>{});
        Tensor gB_nk = local_tile(mB_nk, TileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});

        return cute::make_tuple(gA_mk, gB_nk);
    }

    // Perform a collective-scoped matrix multiply-accumulate
    // Producer Perspective
    template<class TensorA, class TensorB, class KTileIterator, class BlockCoord>
    CUTLASS_DEVICE void
    load(Params const& mainloop_params,
         MainloopPipeline pipeline,
         PipelineState smem_pipe_write,
         cute::tuple<TensorA, TensorB> const& load_inputs,
         BlockCoord const& blk_coord,
         KTileIterator k_tile_iter, int k_tile_count,
         int thread_idx,
         uint32_t block_rank_in_cluster,
         TensorStorage& shared_tensors
    ) {
        int lane_predicate = cute::elect_one_sync();

        if (lane_predicate) {
            Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});
            Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});
            //
            // prepare the tma loads for A and B
            //
            constexpr uint32_t cluster_shape_x = get<0>(typename DispatchPolicy::ClusterShape());
            uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};

            Tensor gA_mk = get<0>(load_inputs);
            Tensor gB_nk = get<1>(load_inputs);

            auto block_tma_a = mainloop_params.tma_load_a.get_slice(cluster_local_block_id.y);
            auto block_tma_b = mainloop_params.tma_load_b.get_slice(cluster_local_block_id.x);

            auto [m_coord, n_coord, k_coord] = blk_coord;
            Tensor gA = gA_mk(_, _, m_coord, _);
            Tensor gB = gB_nk(_, _, n_coord, _);

            Tensor tAgA = block_tma_a.partition_S(gA);
            Tensor tAsA = block_tma_a.partition_D(sA);

            Tensor tBgB = block_tma_b.partition_S(gB);
            Tensor tBsB = block_tma_b.partition_D(sB);

            uint16_t mcast_mask_a = 0;
            uint16_t mcast_mask_b = 0;

            // Mainloop
            CUTLASS_PRAGMA_NO_UNROLL
            for (; k_tile_count > 0; --k_tile_count) {
                pipeline.producer_acquire(smem_pipe_write);

                //
                // copy gmem to smem for *k_tile_iter
                //

                using BarrierType = typename MainloopPipeline::ProducerBarrierType;
                BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

                int write_stage = smem_pipe_write.index();
                copy(mainloop_params.tma_load_a.with(*tma_barrier, mcast_mask_a), tAgA(_, _, _, *k_tile_iter), tAsA(_, _, _, write_stage));
                copy(mainloop_params.tma_load_b.with(*tma_barrier, mcast_mask_b), tBgB(_, _, _, *k_tile_iter), tBsB(_, _, _, write_stage));
                ++k_tile_iter;
                // Advance smem_pipe_write
                ++smem_pipe_write;
            }
        }
    }

    /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
    CUTLASS_DEVICE void
    load_tail(MainloopPipeline pipeline, PipelineState& smem_pipe_write) {
        int lane_predicate = cute::elect_one_sync();

        if (lane_predicate) {
            pipeline.producer_tail(smem_pipe_write);
        }
    }

    /// Perform a collective-scoped matrix multiply-accumulate
    /// Consumer Perspective
    template<class FrgTensorC>
    CUTLASS_DEVICE void
    mma(Params const& mainloop_params,
        MainloopPipeline pipeline,
        PipelineState& smem_pipe_read,
        FrgTensorC& accum,
        int k_tile_count,
        int thread_idx,
        TensorStorage& shared_tensors
    ) {
        static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
        static_assert(cute::rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
        static_assert(cute::rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");
        static_assert(cute::is_void_v<SmemCopyAtomA>,
            "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");
        static_assert(cute::is_void_v<SmemCopyAtomB>,
            "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");
        
        Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});
        Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});

        // 
        // Define C accumulators and A/B partitioning
        //
        // what the fuck
        static_assert(stride<0>(typename TiledMma::ALayout{}) == 0 and
                      stride<0>(typename TiledMma::BLayout{}) == 0 and
                      size<0>(typename TiledMma::ALayout{}) == NumThreadsPerWarpGroup and
                      size<0>(typename TiledMma::BLayout{}) == NumThreadsPerWarpGroup,
                      "Stride of the first mode must be 0 and the size of the mode must be NumThreadsPerWarpGroup");
        
        constexpr int MmaWarpGroups = size(TiledMma{}) / NumThreadsPerWarpGroup;
        Layout warp_group_thread_layout = make_layout(Int<MmaWarpGroups>{}, Int<NumThreadsPerWarpGroup>{});

        int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / NumThreadsPerWarpGroup, 0);

        TiledMma tiled_mma;
        auto thr_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx));

        Tensor tCsA = thr_mma.partition_A(sA); // (mma, m, k, pipe)
        Tensor tCsB = thr_mma.partition_B(sB); // (mma, n, k, pipe)

        // allocate descriptors
        Tensor tCrA = thr_mma.make_fragment_A(tCsA); // (mma, m, k, pipe)
        Tensor tCrB = thr_mma.make_fragment_B(tCsB); // (mma, n, k, pipe)

        CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(accum)); // m
        CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum)); // n
        CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB)); // k
        CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB)); // pipe
        CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));                                         // PIPE
        CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));                                         // PIPE

        // 
        // mainloop
        //
        static_assert((0 <= K_PIPE_MMAS) && (K_PIPE_MMAS <  K_PIPE_MAX), "ERROR : Incorrect number of MMAs in flight");
        
        // We release buffers to producer warps(dma load) with some mmas in flight
        PipelineState smem_pipe_release = smem_pipe_read;

        // Prologue GMMAs
        int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);
        assert(k_tile_count >= 1);
        
        tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
        
        warpgroup_fence_operand(accum);
        
        {
            auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, barrier_token);

            int read_stage = smem_pipe_read.index();
            warpgroup_arrive();

            for (int k_block = 0; k_block < size<2>(tCrA); k_block++) {
                cute::gemm(tiled_mma, tCrA(_, _, k_block, read_stage), tCrB(_, _, k_block, read_stage), accum);
                if (k_block == 0) {
                    tiled_mma.accumulate_ = GMMA::ScaleOut::One;
                } 
            }
            warpgroup_commit_batch();
            ++smem_pipe_read;
        }

        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        warpgroup_fence_operand(accum);

        CUTLASS_PRAGMA_UNROLL
        for (int k_tile_prologue = prologue_mma_count - 1; k_tile_prologue > 0; --k_tile_prologue) {
            auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, barrier_token);

            int read_stage = smem_pipe_read.index();
            warpgroup_arrive();
            cute::gemm(tiled_mma, tCrA(_, _, _, read_stage), tCrB(_, _, _, read_stage), accum);
            warpgroup_commit_batch();

            ++smem_pipe_read;
        }

        warpgroup_fence_operand(accum);

        // mainloop
        k_tile_count -= prologue_mma_count;

        CUTLASS_PRAGMA_NO_UNROLL
        for (; k_tile_count > 0; --k_tile_count) {
            // wait on smem_pipe_read until its data are available
            auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, barrier_token);

            //
            // compute on k_tile
            // 

            int read_stage = smem_pipe_read.index();
            
            warpgroup_fence_operand(accum);
            warpgroup_arrive();
            cute::gemm(tiled_mma, tCrA(_, _, _, read_stage), tCrB(_, _, _, read_stage), accum);
            warpgroup_commit_batch();

            // wait on the gmma barrier for K_PIPE_MMAS
            warpgroup_wait<>(K_PIPE_MMAS);
            warpgroup_fence_operand(accum);

            // unlock smem_pipe_read
            pipeline.consumer_release(smem_pipe_release);

            // Advance smem_pipe_read and smem_pipe_release
            ++smem_pipe_read;
            ++smem_pipe_read;

        }

        warpgroup_fence_operand(accum);
    }

    // Perform a Consumer Epilogue to release all buffers
    mma_tail(MainloopPipeline pipeline, PipelineState smem_pipe_release, int k_tile_count) {
        // prologue gmmas
        int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);
        k_tile_count -= prologue_mma_count;
        smem_pipe_release.advance(k_tile_count);

        // wait on all gmmas to complete
        warpgroup_wait<0>();

        for (int count = 0; count < prologue_mma_count; ++count) {
            pipeline.consumer_release(smem_pipe_release);
            ++smem_pipe_release;
        }
    }

    
    

    
};


    
} // namespace flash
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cute/tensor.hpp>
#include "cutlass/cluster_launch.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/device_kernel.h"
#include "static_switch.h"

using namespace cute;

template <class ElementA,
          class ElementB,
          class ElementC,
          class SmemLayoutA,  // (M,K,P)
          class SmemLayoutB,
          class SmemLayoutC>  // (N,K,P)
struct SharedStorage
{
  alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> smem_A;

  union {
    alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> smem_B;
    alignas(128) cute::ArrayEngine<ElementC, cosize_v<SmemLayoutC>> smem_C;
  };

  uint64_t tma_barrier[size<2>(SmemLayoutA{})];
  uint64_t mma_barrier[size<2>(SmemLayoutA{})];
};

template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA,
          class TB, class SmemLayoutB, class TmaB,
          class TC, class SmemLayoutC, class TmaC,
          class TiledMma>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void gemm_device(ProblemShape shape_MNK,
                 CtaTiler cta_tiler,
                 TA const* A,
                 CUTLASS_GRID_CONSTANT TmaA const tma_a,
                 TB const* B,
                 CUTLASS_GRID_CONSTANT TmaB const tma_b,
                 TC* C,
                 CUTLASS_GRID_CONSTANT TmaC const tma_store,
                 TiledMma mma
) {
    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                   // (M, N, K)
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K)

    static_assert(is_static<SmemLayoutA>::value);
    static_assert(is_static<SmemLayoutB>::value);

    CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutA{}) == size<0>(cta_tiler));  // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutB{}) == size<1>(cta_tiler));  // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutA{}) == size<2>(cta_tiler));  // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutB{}) == size<2>(cta_tiler));  // BLK_K

    auto [M, N, K] = shape_MNK;
    Tensor mA = tma_a.get_tma_tensor(make_shape(M,K));                   // (M,K) TMA Tensor
    Tensor mB = tma_b.get_tma_tensor(make_shape(N,K));
    
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});

    if (thread0()) {
        print("mA: "); print(mA); print("\n");
        print("mB: "); print(mB); print("\n");
        print("gA: "); print(gA); print("\n");
        print("gB: "); print(gB); print("\n");
    }

    extern __shared__ char shared_memory[];

    using SharedStorage = SharedStorage<TA, TB, TC, SmemLayoutA, SmemLayoutB, SmemLayoutC>;
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
    Tensor sA = make_tensor(make_smem_ptr(smem.smem_A.begin()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(smem.smem_B.begin()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

    auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
        group_modes<0,2>(sA), group_modes<0,2>(gA));  // (TMA,k) and (TMA,PIPE)
    
    auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
        group_modes<0,2>(sB), group_modes<0,2>(gB));  // (TMA,k) and (TMA,PIPE)
    
    if (thread0()) {
        print("tAgA: "); print(tAgA); print("\n");
        print("tAsA: "); print(tAsA); print("\n");
        print("tBgB: "); print(tBgB); print("\n");
        print("tBsB: "); print(tBsB); print("\n");
    }
    
    constexpr int kTmaTransactionBytes = CUTE_STATIC_V(size<0>(tAsA)) * sizeof(TA) +
                                         CUTE_STATIC_V(size<0>(tBsB)) * sizeof(TB);
    if (thread0()) {
        printf("kTmaTransactionBytes is %d\n", kTmaTransactionBytes);
    }
    
    auto K_PIPE_MAX = size<1>(tAsA);

    int k_tile_count = size<1>(tAgA);

    int k_tile = 0;
    
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_predicate = cute::elect_one_sync();

    uint64_t* producer_mbar = smem.tma_barrier;
    uint64_t* consumer_mbar = smem.mma_barrier;

    using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;  // TMA
    using ConsumerBarType = cutlass::arch::ClusterBarrier;             // MMA

    for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
        if (warp_idx == 0 && lane_predicate) {
            ProducerBarType::init(&producer_mbar[pipe],   1);
            ConsumerBarType::init(&consumer_mbar[pipe], size(mma));
        }
    }
    // what the fuck
    cluster_sync();

    for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
        if (warp_idx == 0 && lane_predicate) {
            ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], kTmaTransactionBytes);
            copy(tma_a.with(producer_mbar[pipe]), tAgA(_, k_tile), tAsA(_, pipe));
            copy(tma_b.with(producer_mbar[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
        }
        --k_tile_count;
        ++k_tile;
    }

    ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA);                               // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K,PIPE)

    Tensor tCrC = partition_fragment_C(mma, select<0, 1>(cta_tiler));
    clear(tCrC);

    // Allocate descriptor iterators
    Tensor tCrA = thr_mma.make_fragment_A(tCsA);                         // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);                         // (MMA,MMA_N,MMA_K,PIPE)

    if (thread0()) {
        print("tCsA: "); print(tCsA); print("\n");
        print("tCsB: "); print(tCsB); print("\n");
        print("tCrC: "); print(tCrC); print("\n");
        print("tCrA: "); print(tCrA); print("\n");
        print("tCrB: "); print(tCrB); print("\n");
    }

    auto write_state = cutlass::PipelineState<K_PIPE_MAX>();             // TMA writes
    auto read_state  = cutlass::PipelineState<K_PIPE_MAX>();             // MMA  reads

    while (k_tile_count > -K_PIPE_MAX) {
        int read_pipe = read_state.index();
        if (thread0()) {
            printf("read_pipe %d, phase %d\n", read_pipe, read_state.phase());
        }
        ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());
        warpgroup_arrive();
        gemm(mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);     // (V,M) x (V,N) => (V,M,N)
        warpgroup_commit_batch();
        warpgroup_wait<0>();

        ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
        ++read_state;

        if (warp_idx == 0 && lane_predicate && k_tile_count > 0) {
            int pipe = write_state.index();
            ConsumerBarType::wait(&consumer_mbar[pipe], write_state.phase());
            ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], kTmaTransactionBytes);
            copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
            copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));
            ++write_state;
        }
        --k_tile_count;
        ++k_tile;
    }

    if (thread0()) {
        print("tCrC m 0:\n");
        print_tensor(tCrC(_, _0{}, _0{}));
    }

}



template <class T>
void 
wgmma_kernel_launch(const at::Tensor& input,
                    const at::Tensor& weight,
                    at::Tensor& o)
{
    const T* A = reinterpret_cast<T*>(input.data_ptr());
    const T* B = reinterpret_cast<T*>(weight.data_ptr());
    T* C = reinterpret_cast<T*>(o.data_ptr());
    int M = input.size(0);
    int N = weight.size(0);
    int K = input.size(1);
    auto prob_shape = make_shape(M, N, K);

    static constexpr int bM = 256;
    static constexpr int bN = 192;
    static constexpr int bK = 128;
    static constexpr int Stages = 2;
    using AtomLayoutMNK = Layout<Shape<_2, _1, _1>>;

    using cta_tiler = cute::Shape<Int<bM>, Int<bN>, Int<bK>>;
    
    
    // Define the smem layouts (static)
    auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{}, make_shape(Int<bM>{},Int<bK>{},Int<Stages>{}));
    auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{}, make_shape(Int<bN>{},Int<bK>{},Int<Stages>{}));
    auto sC = tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{}, make_shape(Int<bM>{},Int<bN>{}));
    // print("sC: \n");
    // print_layout(sC);
    


    // Define the Tiled MMA
    TiledMMA tiled_mma = make_tiled_mma(SM90_64x192x16_F32F16F16_SS<GMMA::Major::K,GMMA::Major::K>{},
        AtomLayoutMNK{});
    
    


    // using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, T,
    //     decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    // using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));

    // using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
    //     decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    
    // using SmemLayoutK = decltype(tile_to_shape(
    //         SmemLayoutAtomK{},
    //         make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));
    
    // using AtomLayoutQK = Layout<Shape<Int<1>, _1, _1>>;
    // using TiledMmaQK = decltype(make_tiled_mma(
    //             cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShape_MNK>(),
    //             AtomLayoutQK{}));
    
    // using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, head, batch)
    // using StrideQK = cute::Stride<int64_t, _1, int64_t, int64_t>;

    Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), make_stride(K, _1{}));
    Tensor mB = make_tensor(make_gmem_ptr(B), make_shape(N, K), make_stride(K, _1{}));
    Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, _1{}));

    Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(Int<bM>{}, Int<bK>{}));
    Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(Int<bN>{}, Int<bK>{}));
    auto tma_store = make_tma_copy(SM90_TMA_STORE{}, mC, sC, Shape<Int<bM>, Int<bN>>{}, _1{});

    const int smem_size = int(sizeof(SharedStorage<T, T, T, decltype(sA), decltype(sB), decltype(sC)>));

    printf("smem_size is %d\n", smem_size);

    const int num_threads = size(tiled_mma);
    printf("num_threads is %d\n", num_threads);
    print(tiled_mma);
    
    dim3 dimBlock(num_threads);
    dim3 dimCluster(1, 1, 1);
    dim3 dimGrid(round_up(size(ceil_div(M, bM)), dimCluster.x),
               round_up(size(ceil_div(N, bN)), dimCluster.y));
    
    cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};
    
    void const* kernel_ptr = reinterpret_cast<void const*>(
                &gemm_device<decltype(prob_shape), cta_tiler,
                             T, decltype(sA), decltype(tmaA),
                             T, decltype(sB), decltype(tmaB),
                             T, decltype(sC), decltype(tma_store),
                             decltype(tiled_mma)>);
    

    if (smem_size >= 48 * 1024) {
        CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    
    
    cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr,
        prob_shape, cta_tiler{},
        A, tmaA,
        B, tmaB,
        C, tma_store,
        tiled_mma);
    
    CUTE_CHECK_LAST();

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Error: Failed at kernel Launch" << std::endl;
    }
}



void wgmma_matmul(
    const at::Tensor& input,
    const at::Tensor& weight,
    at::Tensor& o
) {

    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm90 = dprops->major == 9;
    TORCH_CHECK(is_sm90, "only supports hopper GPUs.");
    const int smem_size_per_sm = dprops->sharedMemPerMultiprocessor;
    printf("smem_size_per_sm is %d\n", smem_size_per_sm);

    at::cuda::CUDAGuard device_guard{input.device()};
    auto input_dtype = input.dtype();
    TORCH_CHECK(input_dtype == torch::kFloat16 || input_dtype == torch::kBFloat16, "only support fp16 and bf16 data type");
    
    TORCH_CHECK(weight.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(input.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    
    wgmma_kernel_launch<cutlass::half_t>(input, weight, o);
    

}

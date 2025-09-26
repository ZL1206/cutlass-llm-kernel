#include <iostream>
#include <cuda.h>
#include <cute/tensor.hpp>
#include <iostream>
#include <iomanip>
#include <utility>
#include <type_traits>
#include <vector>
#include <numeric>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/trace.h>
#include "cutlass/gemm/collective/builders/sm90_common.inl"



int main(void) {
    using namespace cute;
    using T = cute::half_t;



    using TiledMma = TiledMMA<
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        Layout<Shape<_1, Int<4>, _1>>,  // 4x1x1 or 8x1x1 thread group
        Tile<_16, Layout<Shape <_8,_4,_2>,
                  Stride<_1,_16,_8>>, _16>>;

    using A = Layout< Shape< Shape<Int<2>, Int<2>>, Shape<Int<2>, Int<8>> >,
                     Stride< Stride<Int<2>, Int<4>>, Stride<Int<1>, Int<8>> > >;

    using B = Layout< Shape<Int<128>, Int<64>>,
                     Stride<Int<56>, Int<1>>>;


    using SmemThrLayout = Layout< Shape<Shape<Int<16>, Int<4>>, Int<2>>,
                     Stride<Stride<Int<1>, Int<32>>, Int<16>>>;

    /*
    using SmemThrLayout = Layout< Shape< Shape< Shape<_8, _2>, _4>, _2>,
              Stride< Stride< Stride<_1, _16>, _32>, _8>
              >;
    */
    // AutoVectorizingCopyWithAssumedAlignment<32>

    using SmemTiledCopyKV = decltype(make_tiled_copy(Copy_Atom<SM75_U32x4_LDSM_N, uint8_t>{},
                        SmemThrLayout{},
                        Layout<Shape<_1, _16>>{}));

    //cute::print_latex(TiledMma{});
    auto tiled_copy = make_tiled_copy(Copy_Atom<SM75_U32x4_LDSM_N, uint16_t>{},
                                    Layout<Shape<_32,_1>>{},
                                    Layout<Shape< _1,_8>>{});
    using TiledMma_PV = TiledMMA<
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        Layout<Shape<_1, _1, Int<4>>>,  // 4x1x1 or 8x1x1 thread group
        Tile<_16, _16, Int<16 * 4>>>;
    auto tiled_mma_pv = TiledMma_PV{};
    using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, T>;
    //auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma_pv);

    using TiledMma_qk = TiledMMA<
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        Layout<Shape<Int<4>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        Tile<Int<16 * 4>, _16, _16>>;
    
    using Element = cutlass::half_t;
    using ElementAccum = float;
    static constexpr int kBlockM = 64;
    static constexpr int kBlockN = 16;
    static constexpr int kHeadDim = 128;
    using TileShape_MNK = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using AtomLayoutQK = Layout<Shape<Int<1>, _1, _1>>;
    using TiledMmaQK = decltype(make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShape_MNK>(),
        AtomLayoutQK{}));
    //print_latex(TiledMmaQK{});

    static constexpr int kStages = 1;

    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    printf("SmemLayoutAtomQ: \n");
    print_layout(SmemLayoutAtomQ{});
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));
    printf("SmemLayoutQ: \n");
    print_layout(SmemLayoutQ{});

    auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{}, select<0, 2>(TileShape_MNK{}));
    printf("sA\n");
    print_layout(sA);

    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    printf("SmemLayoutAtomK: \n");
    print_layout(SmemLayoutAtomK{});
    using SmemLayoutK = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}))));
    print("SmemLayoutK: \n");
    print_layout(SmemLayoutK{});
    
    Layout smem_layout = Layout<Shape<_32,_32>, Stride<_32,_1>>{};

    auto cta_tile = product_each(shape(smem_layout));
    print("cta_tile:\n");
    print(cta_tile);
    
    using SmemLayoutShuffle = Layout<Shape<Shape<_1, _2>, Shape<_2, _8, _4>>,
                           Stride<Stride<_0, _8>, Stride<_16, _1, _32>>>;
    printf("SmemLayoutShuffle: \n");
    print_layout(SmemLayoutShuffle{});

    


    using SmemLayoutShuffle_ = Layout<Shape<Shape<_4, _2>, Shape<_4, _4>>,
                           Stride<Stride<_8, _1>, Stride<_2, _32>>>;
    printf("SmemLayoutShuffle_: \n");
    print_layout(SmemLayoutShuffle_{});
    
    const int m = 1024;
    const int k = 1024;
    T* h_A = (T*)malloc(m * k * sizeof(T));
    for (int i = 0; i < m * k; i++) {
        h_A[i] = static_cast<T>(i % 2 == 0 ? 1 : -1);
    }
    
    
    Tensor mA = make_tensor(h_A, make_shape(m, k), make_stride(k, _1{}));

    print(mA);

    auto bM = Int<256>{};
    auto bN = Int<192>{};
    auto bK = Int<128>{};
    auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
    auto cta_coord = make_coord(0, 0, _);              // (m,n,k)
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});
    printf("gA:\n");
    print(gA);
    print("\n");
    print_tensor(gA);

    /*
    print_layout(A{});
    print_layout(B{});


    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<3, 3, 3>{},
                    // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
                    Layout<Shape<_8, Int<64>>,
                           Stride<Int<64>, _1>>{}));

    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<64>, Int<128>>{}));

    using SmemLayoutVtransposed = decltype(
        composition(SmemLayoutKV{}, make_layout(Shape<Int<128>, Int<64>>{}, GenRowMajor{})));

    using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

    print(SmemLayoutVtransposed{});
    printf("\n");
    print(SmemLayoutVtransposedNoSwizzle{});
    print("\n");
    cutlass::half_t* k;
    cudaMalloc(&k, 10 * 256 * 8 * 128 * 2);

    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(k)),
                                make_shape(256, 8, 128),
                                make_stride(1024, 128, _1{}));

    print(mK);
    print("\n");
    Tensor gK = local_tile(mK(_, 0, _), Shape<Int<64>, Int<128>>{},
                           make_coord(0, 0));  // (kBlockM, kHeadDim)
    print(gK);
    */
    using M = Int<32>;
    using N = Int<32>;
    auto block_shape = make_shape(M{}, N{});       // (bM, bN)

    using SmemSwizzle = Swizzle<5, 0, 5>;
    using SmemLayoutAtom = decltype(composition(SmemSwizzle{}, Layout<Shape<Int<8>, Int<32>>, Stride<_32, _1>>{}));
    printf("SmemLayoutAtom:\n");
    print_layout(SmemLayoutAtom{});
    auto tileShapeS = make_layout(block_shape, LayoutRight{});
    using SmemLayout = decltype(composition(SmemSwizzle{}, tileShapeS));

    print_layout(tileShapeS);
    print_layout(SmemLayout{});

    auto block_shape_trans = make_shape(N{}, M{}); // (bN, bM)
    auto tileShapeD = make_layout(block_shape_trans, LayoutRight{});
    auto smemLayoutD_swizzle = composition(SmemLayout{}, tileShapeD);

    print_layout(smemLayoutD_swizzle);

    Layout warp_group_thread_layout = make_layout(Int<2>{},
        Int<128>{});
    printf("warp_group_thread_layout:\n");
    print(warp_group_thread_layout); print("\n");
    for (int i = 0; i < size(warp_group_thread_layout); i++) {
        print(warp_group_thread_layout(i)); print("\n");
    }


    return 0;
}
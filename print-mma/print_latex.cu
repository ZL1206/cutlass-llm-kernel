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
#include "cutlass/detail/helper_macros.hpp"


template<class ElementMma, 
         class AtomLayout = cute::Layout<cute::_1>,
         class ValLayout  = cute::Layout<cute::_1>>
constexpr auto compute_memory_reordering_atom(AtomLayout atom_layout = {}, ValLayout val_layout = {})
{
  using namespace cute;

  static_assert(is_static_v<ValLayout>, "ValLayout must be static");
  static_assert(is_static_v<AtomLayout>, "AtomLayout must be static");

  // 1. Choose an MMA atom to access TV layout and MN shape
  // Note: parameters like GMMA Major, TileShape, ElementC don't affect TV layout of A, use arbitrary
  using MmaAtom = decltype(SM90::GMMA::rs_op_selector<ElementMma, ElementMma, float, Shape<_64,_16,_32>>());
  using MmaTraits = MMA_Traits<MmaAtom>;
  auto mk_shape_mma = select<0,2>(typename MmaTraits::Shape_MNK{});
  auto tv_layout_mma = typename MmaTraits::ALayout{};
  static_assert(size<1>(tv_layout_mma) % size(val_layout) == 0, "Value layout must evenly divide the MMA value layout");

  // 2. Create a single warp's TV layout from that of the whole MMA and invert to get (m,k -> thr,val)
  // Note: this assumes A is partitioned between warps along M mode
  auto tv_tiler_warp = make_shape(Int<32>{}, size<1>(tv_layout_mma));
  auto mk_shape_warp = shape_div(mk_shape_mma, size(typename MmaTraits::ThrID{}) / Int<32>{});
  auto tv_layout_mma_warp = make_layout_like(composition(tv_layout_mma, tv_tiler_warp));
  auto mk_layout_mma_warp = right_inverse(tv_layout_mma_warp).with_shape(mk_shape_warp);

  // 3. Repeat the warp layout NumAtoms times along K mode to get wider vectorization
  auto mk_layout_mma_trgt = blocked_product(mk_layout_mma_warp, atom_layout);

  // 4. Compose with a contiguous layout of values in each thread (required for smem vectorization)
  auto val_to_offset = logical_product(val_layout, size<1>(tv_layout_mma) / size(val_layout) * size(atom_layout));
  auto thr_to_offset = make_layout(size<0>(tv_layout_mma_warp));
  auto tv_to_offset = select<1,0>(logical_product(val_to_offset, thr_to_offset));
  auto layout_atom = composition(tv_to_offset, mk_layout_mma_trgt);

  return layout_atom;
}



int main(void) {
    using namespace cute;
    using T = cute::half_t;
    
    

    using TiledMma = TiledMMA<
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        Layout<Shape<Int<4>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        Tile<Int<16 * 4>, _16, _16>>;
    
    using A = Layout< Shape< Shape<Int<2>, Int<1>>, Shape<Int<2>, Int<2>> >,
                     Stride< Stride<Int<2>, Int<0>>, Stride<Int<1>, Int<4>> > >;

    using B = Layout< Shape<Int<4>, Int<4>>,
                     Stride<Int<1>, Int<8>>>;
    using ValueShuffle = Layout<Shape<_2,_4>, Stride<_4,_1>>;
    
    //print_latex(A{});
    int NumMmaThreads = size(TiledMma{});
    printf("NumMmaThreads is %d\n", NumMmaThreads);
    print_layout(A{});
    print_layout(B{});
    print_layout(ValueShuffle{});


    
    using SmemLayoutAtomKV = decltype(
        composition(Swizzle<2, 3, 3>{},
                    // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
                    Layout<Shape<_8, Int<32>>,
                           Stride<Int<32>, _1>>{}));
    printf("SmemLayoutAtomKV: \n");
    print_layout(SmemLayoutAtomKV{});

    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomKV{},
        Shape<Int<64>, Int<32>>{}));
    print("SmemLayoutKV: \n");
    print_layout(SmemLayoutKV{});
    
    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<3, 3, 3>{},
                    Layout<Shape<_8, Int<64>>,
                           Stride<Int<64>, _1>>{}));
    printf("SmemLayoutAtomQ: \n");
    print_layout(SmemLayoutAtomQ{});

    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<64>, Int<128>>{}));
    
    print("SmemLayoutQ: \n");
    print_layout(SmemLayoutQ{});


    using SmemLayoutQ_ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<64>, Int<128>>{},
        Step<_1,_2>{}));
    
    print("SmemLayoutQ_: \n");
    print_layout(SmemLayoutQ_{});
    
    

    
    using SmemLayoutVtransposed = decltype(
        composition(SmemLayoutKV{}, make_layout(Shape<Int<32>, Int<64>>{}, GenRowMajor{})));

    using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

    print("SmemLayoutVtransposed: \n");
    print_layout(SmemLayoutVtransposed{});
    printf("\n");
    print("SmemLayoutVtransposedNoSwizzle: \n");
    print_layout(SmemLayoutVtransposedNoSwizzle{});
    
    
    // cutlass::half_t* k;
    // cudaMalloc(&k, 10 * 256 * 8 * 128 * 2);

    // Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(k)),
    //                             make_shape(256, 8, 128),
    //                             make_stride(1024, 128, _1{}));
    
    // print(mK);
    // print("\n");
    // Tensor gK = local_tile(mK(_, 0, _), Shape<Int<64>, Int<128>>{},
    //                        make_coord(0, 0));  // (kBlockM, kHeadDim)
    // print(gK);

    // using Th = Layout< Shape<Shape<Int<16>, Int<4>>, Int<2>>,
    //                  Stride<Stride<Int<1>, Int<32>>, Int<16>>>;
    
    // print_layout(Th{});

    // using K = Layout<
    //           Shape< Shape< Shape<_2, _2>, _2>, _8>,
    //           Stride< Stride< Stride<_1, _2>, _4>, _8>
    //           >;
    // print_layout(K{});

    // using M = Layout<
    //           Shape< Shape< Shape<_8, _2>, _4>, _2>,
    //           Stride< Stride< Stride<_1, _16>, _32>, _8>
    //           >;
    // print_layout(M{});

    // //(((_4,_4),_1),_1,_2):(((_1,_64),_0),_0,_32)

    // using P = Layout<
    //           Shape< _64, Shape<_4, _16>>,
    //           Stride< _4, Stride<_1, _256>>
    //           >;
    // print_layout(P{});


    // using SmemLayoutAtomKV4 = Layout<Shape<_8, Int<128>>,
    //                        Stride<Int<128>, _1>>;
  
    // using SmemLayoutKV4 = decltype(tile_to_shape(
    //     SmemLayoutAtomKV4{},
    //     Shape<Int<64>, Int<128>>{}));

    // print_layout(SmemLayoutKV4{});
    // print(cute::cosize_v<SmemLayoutKV>);

    
    // using torvt = Layout<Shape<Shape<_2, _2>, Shape<_8, _2>>,
    //                        Stride<Stride<_1, _2>, Stride<_4, _32>>>;

    // print_layout(torvt{});

    // using Max = Layout<Shape<_4, _8>,
    //                     Stride<_32, _4>>;
    
    // print_layout(Max{});


    // using SmemLayoutAtomO = Layout<Shape<Int<8>, Int<64>>,
    //                        Stride<Int<64>, _1>>;
    // using SmemLayoutO = decltype(tile_to_shape(
    //     SmemLayoutAtomO{},
    //     Shape<Shape<Int<16>, Int<4>>, Int<128>>{}));

    // print_layout(SmemLayoutO{});
    // T* mo;
    // Tensor mO = make_tensor(mo, SmemLayoutO{});    // (SMEM_M,SMEM_N)

    // using GmemLayoutAtom = Layout<Shape <Int<16>, Int<8>>,
    //                               Stride<Int<8>, _1>>;
    
    
    // // write o
    //  using GmemTiledCopyO = decltype(
    //     make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, T>{},
    //                     GmemLayoutAtom{},
    //                     Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store
    // GmemTiledCopyO gmem_tiled_copy_O;
    // auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(0);
    // Tensor tOsO = gmem_thr_copy_O.partition_S(mO);

    // print("tOsO: \n");
    // print(tOsO);


    // using ValueShuffle = Layout<Shape<_2,_4>, Stride<_4,_1>>;
    // T* data = (T*)malloc(8 * sizeof(T));
    // for (int i = 0; i < 8; i++) {
    //     data[i] = i;
    // }
    // Tensor d = make_tensor(data, ValueShuffle{});

    // print_tensor(d);

    // for (int i = 0; i < 8; i++) {
    //     print(d(i)); print("\n");
    // }

    // using MmaType = cutlass::bfloat16_t;
    // using ValueShuffle = Layout<Shape<_2,_4>, Stride<_4,_1>>; // order [0,2,4,6,1,3,5,7]
    // int constexpr NumShuffleAtoms = 1;
    // using MmaAtomShape = Layout<Shape<_1,Int<NumShuffleAtoms>>>;
    // using LayoutAtomQuant = decltype(compute_memory_reordering_atom<MmaType, MmaAtomShape, ValueShuffle>());
    // print_layout(LayoutAtomQuant{});

    // uint16_t* dd = (uint16_t*)malloc(2);
    // dd[0] = 37678;
    // T* aa = reinterpret_cast<T*>(dd);
    // printf("aa 0 is %f\n", static_cast<float>(aa[0]));


    // using K_DQ = Layout< 
    //     Shape<_8, Shape<_2, _2>>,
    //     Stride<_1, Stride<_8, _16>>
    // >;
    // print("K_DQ: \n");
    // print_layout(K_DQ{});
    // printf("layout 8:\n");
    // print(Layout<Shape<_2>, Stride<_1>>{});
    // print("\n");

    // uint16_t* v = (uint16_t*)malloc(128 * 2);
    // for (int i = 0; i < 128; i++) v[i] = i;
    // using RV = Layout<Shape<Shape<_2, _2>, _16>, Stride<Stride<_1, _32>, _2>>;
    // print_layout(RV{});
    // Tensor V = make_tensor(v, RV{});
    // Tensor V_s = V(_, 0);
    // print(rank(V_s.layout())); print("\n");
    // print("V_s: "); print(V_s); print("\n");
    // print_tensor(V_s);

    // Tensor V_s_int32 = cute::recast<uint32_t>(V_s);
    // print("V_s_int32: "); print(V_s_int32); print("\n");
    // print_tensor(V_s_int32);


    // using RV_dq = Layout<Shape<Shape<_8, _2>, _4>, Stride<Stride<_1, _32>, _8>>;
    // print_layout(RV_dq{});
    // Tensor V_dq = make_tensor(v, RV_dq{});
    // print_tensor(V_dq);
    // print(V_dq(make_coord(_, 0), 0));

    // print("\n");
    // print(size<0,0>(V_dq));
    // print("\n");
    // print(size<0,1>(V_dq));
    // print("\n");


    // using SmemLayoutKVData = Layout<Shape<Shape<_1, _4>, _32>,
    //                        Stride<Stride<_0, _8>, _1>>;
    // printf("re: \n");
    // print_layout(SmemLayoutKVData{});

    T* k_data_ptr = (T*)malloc(128 * 2 * 2 * 2);
    for (int i = 0; i < 128; i++) {
        k_data_ptr[i] = i;
        printf("%f", static_cast<float>(k_data_ptr[i])); print(" ");
    }
    // print("\n");
    // Tensor k_data = make_tensor(k_data_ptr, SmemLayoutKVData{});
    // print_tensor(k_data);

    // auto tiled_copy = make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<16>, T>{},
    //                                   Layout<Shape<_2, _32>, Stride<_1, _2>>{},
    //                                   Layout<Shape<_2, _1>>{});

    // auto thr_copy = tiled_copy.get_thread_slice(0);
    // Tensor k_data_t = thr_copy.partition_S(k_data);
    // Tensor tOrO = make_tensor<T>(shape(k_data_t));

    // print("k_data_t: "); print(k_data_t); print("\n");
    // cute::copy(tiled_copy, k_data_t, tOrO);

    // print_tensor(tOrO);

    // float xxx = 9.5;
    // uint8_t yyy = static_cast<uint8_t>(xxx);
    // printf("%d\n", static_cast<uint32_t>(yyy));

    // using SmemLayoutShuffle = Layout<Shape<Shape<_1, _2>, Shape<_2, _8, _4>>,
    //                        Stride<Stride<_0, _8>, Stride<_16, _1, _32>>>;
    // print_layout(SmemLayoutShuffle{});
    // Tensor k_shuffle = make_tensor(k_data_ptr, SmemLayoutShuffle{});

    // Tensor k_first = k_shuffle(_, 0);
    // print_tensor(k_first);
    print("\n");
    using SmemLayoutKVData = Layout<Shape<_2, _4, _16>,
                                    Stride<_64, _16, _1>>;

    // using SmemLayoutKVData = Layout<Shape<_16, _4, _2>,
    //                                 Stride<_1 ,_16, _64>>;

    Tensor k_data = make_tensor(k_data_ptr, SmemLayoutKVData{});
    print_tensor(k_data);


    Tensor k_p = local_tile(k_data(_, 0, _), Shape<_2, _4>{}, make_coord(_0{}, 1));
    print("k_p:\n");
    print_tensor(k_p);

    using SmemLayoutO = Layout<Shape<_8, _8, _2>,
                                    Stride<_16, _1, _8>>;
    Tensor mO = make_tensor(k_data_ptr, SmemLayoutO{});
    Tensor mO_copy = cute::tiled_divide(mO, Shape<_1, Int<4>>{});
    print("mO_copy:\n");
    print(mO_copy);

    constexpr int kBlockMSmem = 8;

    using SmemLSESwizzle = Swizzle<4, 0, 5>;
    using SmemLSESwizzle_ = Swizzle<3, 0, 5>;
    using SmemLayoutAtomLSE =
        decltype(composition(SmemLSESwizzle{},
                 Layout<Shape<Int<8>, Int<kBlockMSmem>>,
                 Stride<Int<kBlockMSmem>, _1>>{}));
    using SmemLayoutAtomLSE_ =
        decltype(composition(SmemLSESwizzle_{},
                 Layout<Shape<Int<8>, Int<kBlockMSmem>>,
                 Stride<Int<kBlockMSmem>, _1>>{}));
    print("SmemLayoutAtomLSE:\n");
    print_layout(SmemLayoutAtomLSE{});
    print_layout(SmemLayoutAtomLSE_{});
    
    
    using SmemLayoutLSE = decltype(tile_to_shape(SmemLayoutAtomLSE{}, Shape<Int<16>, Int<8>>{}));
    print("SmemLayoutLSE: \n");
    print_layout(SmemLayoutLSE{});

    Tensor slse = make_tensor(k_data_ptr, SmemLayoutLSE{});

    print("slse:\n");
    print_tensor(slse);

    static constexpr int MaxThreadsPerBlock = 128;
    static constexpr int AlignmentLSE = std::min(1, int(128 / 8 / sizeof(float)));
    using AlignmentTypeLSE = cute::uint_byte_t<static_cast<int>(sizeof(float)) * AlignmentLSE>;

    static constexpr int kGmemElemsPerLoadLSE = sizeof(AlignmentTypeLSE) / sizeof(float);
    printf("kGmemElemsPerLoadLSE is %d\n", kGmemElemsPerLoadLSE);
    static constexpr int kGmemThreadsPerRowLSE = kBlockMSmem / kGmemElemsPerLoadLSE;
    using GmemLayoutAtomLSE = Layout<Shape <Int<128 / kGmemThreadsPerRowLSE>, Int<kGmemThreadsPerRowLSE>>,
                                     Stride<Int<kGmemThreadsPerRowLSE>, _1>>;
    print("GmemLayoutAtomLSE: \n");
    print_layout(GmemLayoutAtomLSE{});

    using GmemCopyAtomLSE = cute::Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<AlignmentTypeLSE>, float>;
    using GmemTiledCopyLSE = decltype(
        make_tiled_copy(GmemCopyAtomLSE{},
                        GmemLayoutAtomLSE{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoadLSE>>>{}));  // Val layout, 4 vals per load
    
    static constexpr int kSmemThreadsPerColLSEt = MaxThreadsPerBlock / kBlockMSmem;
    using S2RLayoutAtomLSE = Layout<Shape<Int<kSmemThreadsPerColLSEt>, Int<MaxThreadsPerBlock / kSmemThreadsPerColLSEt>>>;

    print("S2RLayoutAtomLSE: \n");
    print(S2RLayoutAtomLSE{});
    print("\n");
    using S2RTiledCopyLSE = decltype(make_tiled_copy(cute::Copy_Atom<cute::DefaultCopy, float>{}, S2RLayoutAtomLSE{}, Layout<_1>{}));

    Tensor sLSE = make_tensor(k_data_ptr, SmemLayoutLSE{});

    S2RTiledCopyLSE s2r_tiled_copy_LSE;
    auto s2r_thr_copy_LSE = s2r_tiled_copy_LSE.get_thread_slice(0);
    Tensor ts2rsLSE = s2r_thr_copy_LSE.partition_S(sLSE);

    print("ts2rsLSE: "); print(ts2rsLSE); print("\n");

    Tensor cLSE = make_identity_tensor(make_shape(32, 8));
    
    Tensor ts2rcLSE = s2r_thr_copy_LSE.partition_D(cLSE);
    print("ts2rcLSE: \n");
    print_tensor(ts2rcLSE);


    using GmemCopyAtomLSE = cute::Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<AlignmentTypeLSE>, float>;

    using GmemTiledCopyLSE = decltype(
        make_tiled_copy(GmemCopyAtomLSE{},
                        GmemLayoutAtomLSE{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoadLSE>>>{}));  // Val layout, 4 vals per load

    
    GmemTiledCopyLSE gmem_tiled_copy_LSE;
    auto gmem_thr_copy_LSE = gmem_tiled_copy_LSE.get_thread_slice(0);

    

    Tensor tLSEcLSE = gmem_thr_copy_LSE.partition_S(cLSE);

    print("tLSEcLSE: \n");
    print_tensor(tLSEcLSE);


    cutlass::FastDivmod seqlen_divmod_dynamic(1);

    int m_idx;
    int h_idx = seqlen_divmod_dynamic.divmod(m_idx, 7);
    print("m is %d, h is %d\n", m_idx, h_idx);
    
    using GmemLayoutAtom = Layout<Shape <Int<8>, Int<16>>,
                                  Stride<Int<16>, _1>>;
    
    using GmemCopyAtom = cute::Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, cutlass::half_t>;
    using GmemTiledCopyAccum = decltype(
        make_tiled_copy(GmemCopyAtom{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<8>>>{}));  // Val layout, 4 vals per load
    
    GmemTiledCopyAccum gmem_tiled_copy_O_partial;
    auto gmem_thr_copy_O_partial = gmem_tiled_copy_O_partial.get_thread_slice(0);
    using TileShape_MK = cute::Shape<Int<32>, Int<128>>;
    Tensor cO = cute::make_identity_tensor(TileShape_MK{});  // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Layout tOrOpartial_layout = gmem_thr_copy_O_partial.partition_S(make_tensor<cutlass::half_t>(TileShape_MK{})).layout();
    Layout tOcO = gmem_thr_copy_O_partial.partition_D(cO).layout();

    print("tOrOpartial_layout: "); print(tOrOpartial_layout); print("\n");
    print("tOcO: "); print(tOcO); print("\n");
    Tensor tOrOpartial = make_fragment_like<cutlass::half_t>(tOrOpartial_layout);
    Tensor tOrO = make_fragment_like<float>(tOrOpartial);

    print("tOrOpartial: "); print(tOrOpartial); print("\n");

    
    print("what is this:\n");
    uint32_t line = 1;
    CUTLASS_UNUSED(line);

    auto t_tiled_copy = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, float>{},
                                           Layout<Shape<_16, _8>, Stride<_8, _1>>{},
                                           Layout<Shape<_1, _4>>{});
    print("t_tiled_copy\n");
    print(t_tiled_copy);
    
    float* t_data_ptr = (float*)malloc(64 * 64 * 4);
    Tensor t_tensor = make_tensor(t_data_ptr, make_shape(64, 64), make_stride(64, _1{}));
    auto thr_tiled_copy = t_tiled_copy.get_slice(0);
    Tensor gt = thr_tiled_copy.partition_S(t_tensor);

    print("gt: "); print(gt); print("\n");
    
    return 0;
}


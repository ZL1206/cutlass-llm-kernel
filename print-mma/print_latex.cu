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



int main(void) {
    using namespace cute;
    using T = cute::half_t;
    
    

    using TiledMma = TiledMMA<
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        Layout<Shape<Int<4>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        Tile<Int<16 * 4>, _16, _16>>;
    
    using A = Layout< Shape< Shape<Int<2>, Int<2>>, Shape<Int<2>, Int<8>> >,
                     Stride< Stride<Int<2>, Int<4>>, Stride<Int<1>, Int<8>> > >;

    using B = Layout< Shape<Int<4>, Int<4>>,
                     Stride<Int<1>, Int<8>>>;
    
    //print_latex(A{});
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
    return 0;
}


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

    using B = Layout< Shape<Int<128>, Int<64>>,
                     Stride<Int<56>, Int<1>>>;
    
    //print_latex(A{});
    print_layout(A{});
    print_layout(B{});
    
    
    return 0;
}


#include <iostream>
#include <cuda.h>
#include <cute/tensor.hpp>
#include <iostream>
#include <iomanip>
#include <utility>
#include <type_traits>
#include <vector>
#include <numeric>
#include <random>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/trace.h>


using namespace cute;

template<typename Kernel_traits>
__global__ void g2rCopy(void* q) {
    
    using T = typename Kernel_traits::T;
    using GmemTiledCopyQKV = typename Kernel_traits::GmemTiledCopyQKV;

    constexpr int M = Kernel_traits::kTileM;
    constexpr int N = Kernel_traits::kTileN;
    constexpr int K = Kernel_traits::kTileK;



    Tensor gQ = make_tensor(make_gmem_ptr((T *)q),
                    make_shape(Int<M>{}, Int<K>{}),
                    make_stride(Int<K>{}, Int<1>{}));

    int idx = threadIdx.x;

    GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_slice(idx);
    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    
    auto fragment = make_fragment_like(tQgQ);

    if (thread0()){
        print("gmem_tiled_copy_QKV: \n"); print(gmem_tiled_copy_QKV); print("\n");
        print("tQgQ: \n"); print(tQgQ); print("\n");
        print("fragment: \n"); print(fragment); print("\n");
        print("size 0: "), print(size<0>(tQgQ)); print("\n");
        print("size 1: "), print(size<1>(tQgQ)); print("\n");
        print("size 2: "), print(size<2>(tQgQ)); print("\n");

    }
    for (int m = 0; m < size<1>(tQgQ); m++) {
        for (int k = 0; k < size<2>(tQgQ); k++) {
            if (thread0()) {
                printf("m %d, k %d \n", m, k); print_tensor(tQgQ(_, m, k));
            }
            copy(gmem_tiled_copy_QKV, tQgQ(_, m, k), fragment(_, m, k));
        }
    }
    
    if (thread0()) {
        printf("tensor fragment:\n");
        print_tensor(fragment);
        printf("Thread ID: %d \n", idx);
        for (int i = 0; i < fragment.size(); i++) {
            float tmp = static_cast<float>(fragment(i));
            printf("%10.6f ", tmp);
            if ((i + 1) % 8 == 0) {
                print("\n");       
            }
        }
    }   
}


template <typename T_, int kTileM_ = 128, int kTileN_ = 32, int kTileK_ = 128>
struct Kernel_traits {

  using T = T_;

  // tile configuration
  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kTileK = kTileK_;
  
  
  // global to register
  using GmemLayoutAtom = Layout<Shape <Int<16>, Int<8>>,
                                  Stride<Int<8>, _1>>;
    
  using GmemTiledCopyQKV = decltype(make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, T>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));
};


int main(void) {
    using namespace cute;
    using T = cute::half_t;
    
    constexpr int M = 128;
    constexpr int N = 64;
    constexpr int K = 128;

    std::mt19937 gen(20250102);
    std::uniform_real_distribution<float> dis(static_cast<float>(-100), static_cast<float>(100));
    
    T* h_q = (T*)malloc(M * K * sizeof(T));
    for (int i = 0; i < M * K; i++) {
        float data = dis(gen);
        h_q[i] = T(data);
        printf("%10.6f", data); printf(" ");
        if ((i + 1) % 8 == 0 && (i+1) % 128 != 0) {
            printf("|"); printf(" ");
        }
        if ((i+1) % 128 == 0) {
            printf("\n");
        }
    }
    T* d_q = nullptr;
    cudaMalloc(&d_q, M * K * sizeof(T));
    cudaMemcpy(d_q, h_q, sizeof(T) * M * K, cudaMemcpyHostToDevice);

    
    Kernel_traits<T, M, N, K> config;

    auto kernel = &g2rCopy<decltype(config)>;

    kernel<<<1, 128>>>(d_q);

    auto err = cudaGetLastError();
    printf("Copy done, Error Code: %d, State: %s\n", err, cudaGetErrorString(err));

    
    cudaFree(d_q);
    
    return 0;
}
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>
#include "cutlass/util/command_line.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "cutlass/detail/layout.hpp"
#include <random>


using namespace cute;


struct Kernel_traits {

  using T = float;

  static constexpr int kDim = 64;

  using Tile = Layout<Shape<Int<kDim>, Int<kDim>>, Stride<Int<kDim>, _1>>;
   
  using SmemLayout = Tile;
  //using SmemLayout = Tile;

  static constexpr int kNWarps = 4;
  static constexpr int kNThreads = kNWarps * 32;

  using GmemLayoutAtom = Layout<Shape<Int<16>, Int<8>>,
                                Stride<Int<8>, _1>>;
  
  using GmemTiledCopy = decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, float>{},
                                                  GmemLayoutAtom{},
                                                  Layout<Shape<_1, _4>>{}));
  
  struct TensorStorage {
    cute::array_aligned<float, cute::cosize_v<SmemLayout>> smem;
  };

  static constexpr int kSmemSize = sizeof(TensorStorage);

};

template<typename Kernel_traits>
__global__ void copy(void* input, void* out, const int m, const int n) {

  int tid = threadIdx.x;

  using T = typename Kernel_traits::T;
  constexpr int kDim = Kernel_traits::kDim;

  using SharedStorage = typename Kernel_traits::TensorStorage;
  extern __shared__ char smem_[];
  SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_);

  Tensor mA = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(input)), make_shape(m, n), make_stride(n, _1{}));

  Tensor gA = local_tile(mA, Shape<Int<kDim>, Int<kDim>>{}, make_coord(blockIdx.x, blockIdx.y));

  Tensor s_tile = make_tensor(make_smem_ptr(shared_storage.smem.data()), typename Kernel_traits::SmemLayout{});

  
  typename Kernel_traits::GmemTiledCopy gmem_tiled_copy;
  auto gmem_thr_copy = gmem_tiled_copy.get_slice(tid);
  Tensor tg = gmem_thr_copy.partition_S(gA);
  Tensor ts = gmem_thr_copy.partition_D(s_tile);
  cute::copy(gmem_tiled_copy, tg, ts);
  
  cute::cp_async_fence();
  cute::cp_async_wait<0>();
  __syncthreads();

  if (thread0()) {
    print("tg: "); print(tg); print("\n");
    print("ts: "); print(ts); print("\n");
  }
  if (thread0()) {
    print_tensor(s_tile);
  }
  
//   typename Kernel_traits::S2RTiledCopy s2r_tiled_copy;
//   auto s2r_thr_copy = s2r_tiled_copy.get_slice(tid);
//   Tensor ts2rs = s2r_thr_copy.partition_S(s_tile);
//   Tensor ts2rr = make_fragment_like(ts2rs);
//   copy(s2r_tiled_copy, ts2rs, ts2rr);

//   Tensor dst = out(make_coord(_, _), blockIdx.y, blockIdx.x);
//   Tensor td = gmem_thr_copy.partition_D(dst);

//   for (int mi = 0; mi < size<1>(td); mi++) {
//     for (int ki = 0; ki < size<2>(td); ki++) {
//       copy(gmem_tiled_copy, ts2rr(_, ki, mi), td(_, mi, ki));
//     }
//   }
}


void transpose_smem(float* input_, float* out_, int m, int n) {

  Kernel_traits config;
  
  Tensor input = make_tensor(make_gmem_ptr(input_), make_shape(m, n), make_stride(n, _1{}));
  Tensor out = make_tensor(make_gmem_ptr(out_), make_shape(m, n), make_stride(n, _1{}));
  
  constexpr int kDim = Kernel_traits::kDim;
  
  Tensor tiled_input = tiled_divide(input, Shape<Int<kDim>, Int<kDim>>{});
  Tensor tiled_out = tiled_divide(out, Shape<Int<kDim>, Int<kDim>>{});
  print("tiled_input:\n");
  print(tiled_input); print("\n");
  const int smem_size = config.kSmemSize;

  auto kernel = &copy<decltype(config)>;

  if (smem_size >= 48 * 1024) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  }


  dim3 gridDim(
    size<1>(tiled_input),
    size<2>(tiled_input)); 
  
  kernel<<<gridDim, config.kNThreads, smem_size>>>(input_, out_, m, n);
    
}


int main(int argc, char const **argv) {

    cutlass::CommandLine cmd(argc, argv);
    using T = float;
    using namespace cute;

    int M, N;
    cmd.get_cmd_line_argument("M", M, 256);
    cmd.get_cmd_line_argument("N", N, 256);

    std::cout << "Matrix size: " << M << " x " << N << std::endl;

    
    T* h_input = (T*)malloc(M * N * sizeof(T));
    T* d_input;
    cudaMalloc(&d_input, M * N * sizeof(T));
    
    T* h_out = (T*)malloc(M * N * sizeof(T));
    T* d_out;
    cudaMalloc(&d_out, M * N * sizeof(T));
    
    std::mt19937 gen(0);
    std::uniform_int_distribution<int> dis(1, 100);
    for (size_t i = 0; i < M * N; ++i) {
      h_input[i] = static_cast<T>(i % N);
      // h_input[i] = static_cast<T>(dis(gen));
    }

    cudaMemcpy(d_input, h_input, N * M * sizeof(T), cudaMemcpyHostToDevice);

    transpose_smem(d_input, d_out, M, N);

    cudaMemcpy(h_out, d_out, N * M * sizeof(T), cudaMemcpyDeviceToHost);


    cudaDeviceSynchronize();
    // auto err = cudaGetLastError();
    // printf("Copy done, Error Code: %d, State: %s\n", err, cudaGetErrorString(err));

    // bool pass = true;
    // for (int n = 0; n < N; n++) {
    //   for (int m = 0; m < M; m++) {
    //     float data = h_input[m * N + n];
    //     float transpose_data = h_out[n * M + m];
    //     if (transpose_data != data) {
    //       pass = false;
    //     }
    //     printf("%12f ", transpose_data); print(" ");
    //   }
    //   printf("\n");
    // }

    // if (!pass) {
    //   printf("fuck\n");
    // } else {
    //   printf("pass\n");
    // }



    

    return 0;
}
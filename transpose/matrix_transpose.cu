#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>
#include "cutlass/util/command_line.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

template <typename T> struct TransposeParams {
  T *input;
  T *output;

  const int M;
  const int N;

  TransposeParams(T *input_, T *output_, int M_, int N_)
      : input(input_), output(output_), M(M_), N(N_) {}
};



template <class TensorS, class TensorD, class ThreadLayout, class VecLayout>
__global__ static void __launch_bounds__(256, 1) copyKernel(TensorS const S, TensorD const D, ThreadLayout, VecLayout) {
    using namespace cute;
    using Element = typename TensorS::value_type;

    Tensor gS = S(make_coord(_, _), blockIdx.x, blockIdx.y);   // (bM, bN)
    Tensor gD = D(make_coord(_, _), blockIdx.x, blockIdx.y); // (bN, bM)

    // Define `AccessType` which controls the size of the actual memory access.
    using AccessType = cutlass::AlignedArray<Element, size(VecLayout{})>;

    using Atom = Copy_Atom<UniversalCopy<AccessType>, Element>;

    auto tiled_copy = make_tiled_copy(
        Atom{},                       // access size
        ThreadLayout{},               // thread layout
        VecLayout{});                 // vector layout (e.g. 4x1)
    

    auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);

    Tensor tSgS = thr_copy.partition_S(gS);             // (CopyOp, CopyM, CopyN)
    Tensor tDgD = thr_copy.partition_D(gD);             // (CopyOp, CopyM, CopyN)

    Tensor rmem = make_tensor_like(tSgS);               // (ThrValM, ThrValN)

    copy(tSgS, rmem);
    copy(rmem, tDgD);



}

template <typename T>
void copy_baseline(TransposeParams<T> params) {
    using Element = float;
    using namespace cute;
    Shape<int, int> tensor_shape = make_shape(params.M, params.N);
    Layout gmemLayoutS = make_layout(tensor_shape, LayoutRight{});
    Layout gmemLayoutD = make_layout(tensor_shape, LayoutRight{});
    Tensor tensor_S = make_tensor(make_gmem_ptr(params.input), gmemLayoutS);
    Tensor tensor_D = make_tensor(make_gmem_ptr(params.output), gmemLayoutD);

    //
    // Tile tensors
    //
    using bM = Int<32>;
    using bN = Int<32>;

    auto block_shape = make_shape(bM{}, bN{});       // (bM, bN)

    Tensor tiled_tensor_S = tiled_divide(tensor_S, block_shape); // ((bM, bN), m', n')
    Tensor tiled_tensor_D = tiled_divide(tensor_D, block_shape); // ((bN, bM), n', m')
    print("tiled_tensor_S"); print(tiled_tensor_S); print("\n");
    print("tiled_tensor_D"); print(tiled_tensor_D); print("\n");
    auto threadLayout =
      make_layout(make_shape(Int<8>{}, Int<32>{}), LayoutRight{});

    auto vec_layout = make_layout(make_shape(Int<4>{}, Int<1>{}));

    dim3 gridDim(
      size<1>(tiled_tensor_S),
      size<2>(tiled_tensor_S)); // Grid shape corresponds to modes m' and n'
    dim3 blockDim(size(threadLayout)); // 256 threads

    copyKernel<<<gridDim, blockDim>>>(tiled_tensor_S, tiled_tensor_D,
                                       threadLayout,  vec_layout);




}


int main(int argc, char const **argv) {

    cutlass::CommandLine cmd(argc, argv);
    using T = float;
    using namespace cute;

    int M, N;
    cmd.get_cmd_line_argument("M", M, 256);
    cmd.get_cmd_line_argument("N", N, 256);

    std::cout << "Matrix size: " << M << " x " << N << std::endl;

    auto tensor_shape_S = make_shape(M, N);

    auto tensor_shape_D = make_shape(N, M);

    thrust::host_vector<T> h_S(size(tensor_shape_S));

    thrust::host_vector<T> h_D(size(tensor_shape_D)); // (N, M)

    for (size_t i = 0; i < h_S.size(); ++i) h_S[i] = static_cast<T>(i);

    thrust::device_vector<T> d_S = h_S;
    thrust::device_vector<T> d_D = h_D;

    TransposeParams<T> params(thrust::raw_pointer_cast(d_S.data()), thrust::raw_pointer_cast(d_D.data()), M, N);

    Layout transpose_function = make_layout(tensor_shape_S, LayoutRight{});

    // just copy, no transpose
    copy_baseline(params);

    h_D = d_D;

    int bad = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
          if (h_D[i * N + j] != h_S[i * N + j]) {
            bad++;
          }
        }
    }

    if (bad > 0) {
      std::cout << "Validation failed. Correct values: " << h_D.size()-bad << ". Incorrect values: " << bad << std::endl;
    } else {
      std::cout << "Validation success." << std::endl;
    }



     


    return 0;
}
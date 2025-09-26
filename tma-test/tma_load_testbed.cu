#include <iostream>
#include <cstdint>
#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>



using namespace cute;

template <class ElementType, class SmemLayout>
struct SharedStorage
{
  cute::ArrayEngine<ElementType, cute::cosize_v<SmemLayout>> smem;
  alignas(16) cute::uint64_t tma_load_mbar[1];
};

template <class T, class TiledCopy, class CTA_Tiler, class GmemLayout, class SmemLayout>
__global__ void
tma_test_device_cute(const T* g_in,
                     CUTE_GRID_CONSTANT TiledCopy const tma, CTA_Tiler cta_tiler,
                     GmemLayout gmem_layout, SmemLayout smem_layout) 
{
    CUTE_STATIC_ASSERT_V(product_each(shape(cta_tiler)) == product_each(shape(smem_layout)));
    extern __shared__ char shared_memory[];
    using SharedStorage = SharedStorage<T, SmemLayout>;
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);
    Tensor sA = make_tensor(make_smem_ptr(shared_storage.smem.begin()), smem_layout);  // (CTA_TILE_M,CTA_TILE_N,...)
    uint64_t* tma_load_mbar = shared_storage.tma_load_mbar;

    auto g_shape = shape(gmem_layout);
    Tensor mA = tma.get_tma_tensor(shape(gmem_layout));
    Tensor gA = local_tile(mA, Shape<Int<32>, Int<32>>{}, make_coord(_0{},_0{}));

    auto cta_tma = tma.get_slice(Int<0>{});                            // CTA slice
    
    Tensor tAgA_x = cta_tma.partition_S(gA);                           
    Tensor tAsA_x = cta_tma.partition_D(sA);

    Tensor tAgA = group_modes<0, 3>(tAgA_x);
    Tensor tAsA = group_modes<0, 3>(tAsA_x);
    
    constexpr int R = rank_v<CTA_Tiler>;
    
    if (thread0()) {
        print(tma);
        print("R: "); print(R); print("\n");
        print("TILE  :  "); print(cta_tiler); print("\n");
        print("  mA  :  "); print(  mA);   print("\n");
        print("  g_shape  :  "); print(  g_shape);   print("\n");
        print("  gA  :  "); print(  gA);   print("\n");
        print_tensor(gA);
        print("  sA  :  "); print(  sA);   print("\n");
        print("tAgA_x:  "); print(tAgA_x); print("\n");
        
        print("tAsA_x:  "); print(tAsA_x); print("\n");
        print("tAgA  :  "); print(tAgA); print("\n");
        print("tAsA  :  "); print(tAsA); print("\n");
    }

    // // Test L2 prefetch
    // if (threadIdx.x == 0) {
    //     prefetch(tma, tAgA);
    // }



    constexpr int TmaTransactionBytesQ = size(gmem_layout) * sizeof(T);

    if (threadIdx.x == 0) {
        cute::initialize_barrier(tma_load_mbar[0], 1 /*numThreads*/);
        cute::set_barrier_transaction_bytes(tma_load_mbar[0], TmaTransactionBytesQ);
        copy(tma.with(tma_load_mbar[0]), tAgA_x, tAsA_x);
    }
    
    int kPhaseBit = 0;
    cute::wait_barrier(tma_load_mbar[0], kPhaseBit);

    
    // for (int stage = 0; stage < size<1>(tAgA); ++stage) {
    //     constexpr int kTmaTransactionBytes = sizeof(make_tensor_like(tensor<0>(tAsA)));

    //     if (threadIdx.x == 0) {
    //         tma_load_mbar[0] = 0;
    //         cute::initialize_barrier(tma_load_mbar[0], 1 /*numThreads*/);
    //         cute::set_barrier_transaction_bytes(tma_load_mbar[0], kTmaTransactionBytes);
    //         copy(tma.with(tma_load_mbar[0]), tAgA(_,stage), tAsA(_,0));
    //     }
    //     __syncthreads();

    //     constexpr int kPhaseBit = 0;
    //     cute::wait_barrier(tma_load_mbar[0], kPhaseBit);
    // }

    __syncthreads();
    if (thread0()) {
        print("sA: \n");
        print_tensor(sA);
        printf("kPhaseBit is %d\n", kPhaseBit);
    }



}





int main () { 
    
    using CopyOp = cute::SM90_TMA_LOAD;
    using T = cutlass::half_t;

    Layout smem_layout = Layout<Shape<_32,_32>, Stride<_32,_1>>{};
    Layout gmem_layout = smem_layout;
    auto cta_tile = product_each(shape(smem_layout));

    const int size = cosize(gmem_layout);

    printf("size is %d\n", size);

    T* h_in = (T*)malloc(size * sizeof(T));
    for (int i = 0; i < size; i++) {
        h_in[i] = static_cast<T>(i);
    }
    T* d_in;
    cudaMalloc(&d_in, size * sizeof(T));
    
    cudaMemcpy(d_in, h_in, sizeof(T) * size, cudaMemcpyHostToDevice);

    Tensor gA = make_tensor(make_gmem_ptr(d_in), gmem_layout);

    auto tma = make_tma_copy<T>(CopyOp{}, gA, smem_layout, cta_tile, Int<1>{});

    int smem_size = int(sizeof(SharedStorage<T, decltype(smem_layout)>));

    tma_test_device_cute<<<1, 128, smem_size>>>(d_in, tma, cta_tile, gmem_layout, smem_layout);

    cudaDeviceSynchronize();
    // auto err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,
    //          cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }

    return 0;

}




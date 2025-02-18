#include <cuda/atomic>

#include <cuda_runtime.h>
#include <iostream>

#define THREADS_PER_BLOCK 256

__global__ void atomic_add_kernel(int* data, int* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 使用全局内存地址初始化 cuda::atomic_ref 对象
    cuda::atomic_ref<int, cuda::thread_scope_device> atomic_data(*data);

    // 执行原子加法操作并返回操作前的值
    int old_value = atomic_data.fetch_add(1, cuda::memory_order_relaxed);
    printf("tid is %d, old value is %d \n", idx, old_value);
    // 将返回值存储到 result 数组
    result[idx] = old_value;
}

int main() {
    const int num_threads = 256;
    int h_data = 0;
    int* d_data;
    int* d_result;
    int h_result[num_threads];

    // 分配 GPU 内存
    cudaMalloc(&d_data, sizeof(int));
    cudaMalloc(&d_result, num_threads * sizeof(int));

    cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice);

    // 启动内核
    atomic_add_kernel<<<num_threads / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_data, d_result);

    // 将结果从设备内存复制回主机
    cudaMemcpy(&h_result, d_result, num_threads * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost);

    // 输出结果
    std::cout << "Final value: " << h_data << std::endl;
    
    // 清理 GPU 内存
    cudaFree(d_data);
    cudaFree(d_result);

    return 0;
}
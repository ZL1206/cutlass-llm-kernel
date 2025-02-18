#include <iostream>
#include <cuda_runtime.h>

__global__ void simpleKernel(int* d_data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    d_data[idx] *= 2; // 简单操作，确保内核执行
}

int main() {
    const int dataSizeSmall = 16 * 1024; // 16 KB
    const int dataSizeLarge = 256 * 1024; // 128 KB

    int* h_dataSmall = new int[dataSizeSmall / sizeof(int)];
    int* h_dataLarge = new int[dataSizeLarge / sizeof(int)];

    int* d_dataSmall;
    int* d_dataLarge;

    cudaMalloc(&d_dataSmall, dataSizeSmall);
    cudaMalloc(&d_dataLarge, dataSizeLarge);


    // 这段是小于64KB的数据传输
    cudaMemcpy(d_dataSmall, h_dataSmall, dataSizeSmall, cudaMemcpyHostToDevice);
    simpleKernel<<<1, 256>>>(d_dataSmall); // 在设备上执行一个简单核函数


    // 这段是大于64KB的数据传输
    cudaMemcpy(d_dataLarge, h_dataLarge, dataSizeLarge, cudaMemcpyHostToDevice);
    simpleKernel<<<1, 256>>>(d_dataLarge); // 在设备上执行同样的核函数



    cudaFree(d_dataSmall);
    cudaFree(d_dataLarge);
    delete[] h_dataSmall;
    delete[] h_dataLarge;

    return 0;
}


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

#define TOLERANCE 0.00001
__host__ int retCores(cudaDeviceProp *device) {
    return 128 * device->multiProcessorCount; 
}

__global__ void matrixMultiplicationKernel(float* P, const float* M, const float* N, int numDimensions) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        for (int row = 0; row < numDimensions; row++) {
            for (int col = 0; col < numDimensions; col++) {
                float sum = 0.0f;
                for (int i = 0; i < numDimensions; i++) {
                    sum += M[row * numDimensions + i] * N[i * numDimensions + col];
                }
                P[row * numDimensions + col] = sum;
            }
        }
    }
}


__host__ void matrixMultiplicationCpu(float *P, const float *M, const float *N, int numDimensions) {
    for (int i = 0; i < numDimensions; i++) {
        for (int j = 0; j < numDimensions; j++) {
            float sum = 0.0f;

            for (int k = 0; k < numDimensions; k++) {
                sum += M[i * numDimensions + k] * N[k * numDimensions + j];
            }
            P[i * numDimensions + j] = sum;
        }
    }
}

void verifyMatrix(float *matrix1, float *matrix2, int numDimensions) {
    for (int i = 0; i < numDimensions * numDimensions; i++) {
        if (fabs(matrix1[i] - matrix2[i]) > TOLERANCE) {
            printf("Test FAILED\n");
        }
    }
    printf("Test PASSED\n");
}

void initializeMatrix(float* matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main() {

    //part1 
    // device : NVIDIA 3060ti. device major = 8, according to online documentation this is an ampere convention which means 128 * the mp count. 
    int nd;
    int n = 512; // 512 ctrl shift b -> ctrl f5

    //host -> device

    cudaGetDeviceCount(&nd);
    for (int i = 0; i < nd; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        int cores = retCores(&deviceProps);
        printf("Number of devices: %d\n CUDA device [%s]\n Clock Rate: %d \n MultiProcessorCount: %d\n Number of Cores: %d\n", nd, deviceProps.name, deviceProps.clockRate, deviceProps.multiProcessorCount, cores);
        printf(" warp size: %d\n total global memory: %zu \n shared mem per block %zu \n num registers per block %d \n max threads per block %d \n max dimension size of each block (%d, %d, %d) \n max size of grid (%d, %d, %d) \n",
            deviceProps.warpSize, deviceProps.totalGlobalMem, deviceProps.sharedMemPerBlock, deviceProps.regsPerBlock, deviceProps.maxThreadsPerBlock, deviceProps.maxThreadsDim[0],
            deviceProps.maxThreadsDim[1], deviceProps.maxThreadsDim[2], deviceProps.maxGridSize[0], deviceProps.maxGridSize[1], deviceProps.maxGridSize[2]);
    }
    //part2
    size_t size = n * n * sizeof(float);

    //alloc host mem
    float* h_M = (float*)malloc(size);
    float* h_N = (float*)malloc(size);
    float* h_P = (float*)malloc(size);
    float* h_P_cpu = (float*)malloc(size);

    //alloc device mem
    float* d_M, * d_N, * d_P;
    cudaMalloc(&d_M, size);
    cudaMalloc(&d_N, size);
    cudaMalloc(&d_P, size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();

    dim3 threadsPerBlock(1, 1); // 256 threads
    dim3 blocksPerGrid(1, 1);
    //dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    float gpu_time = 0.0f;
    float cpu_time = 0.0f;

    initializeMatrix(h_M, n * n);
    initializeMatrix(h_N, n * n);

    //part 2 matrix mult 
    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
    cudaEventRecord(start, 0);
    matrixMultiplicationKernel<<<blocksPerGrid, threadsPerBlock>>> (d_P, d_M, d_N, n);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);

    cudaEventRecord(start, 0);
    matrixMultiplicationCpu(h_P, h_M, h_N, n);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_time, start, stop);

    cudaMemcpy(h_P_cpu, d_P, size, cudaMemcpyDeviceToHost);
    verifyMatrix(h_P, h_P_cpu, n);

    printf("CPU TIME: %f\n GPU TIME: %f\n", cpu_time, gpu_time);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    free(h_M);
    free(h_N);
    free(h_P);
    free(h_P_cpu);



    //part 1 - data collection 
    //float total_H2D = 0.0f, total_D2H = 0.0f;
    //for (int i = 0; i < 6; i++) {
    //    float gpu_time = 0.0f;

    //    // host to Device Transfer Time
    //    cudaEventRecord(start, 0);
    //    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    //    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
    //    cudaEventRecord(stop, 0);
    //    cudaEventSynchronize(stop);
    //    cudaEventElapsedTime(&gpu_time, start, stop);
    //    total_H2D += gpu_time;
    //    printf("Run %d - Host to Device Transfer Time: %f ms\n", i + 1, gpu_time);

    //    // device to Host Transfer Time
    //    cudaEventRecord(start, 0);
    //    cudaMemcpy(h_M, d_M, size, cudaMemcpyDeviceToHost);
    //    cudaMemcpy(h_N, d_N, size, cudaMemcpyDeviceToHost);
    //    cudaEventRecord(stop, 0);
    //    cudaEventSynchronize(stop);
    //    cudaEventElapsedTime(&gpu_time, start, stop);
    //    total_D2H += gpu_time;
    //    printf("run %d - Device to Host Transfer Time: %f ms\n", i + 1, gpu_time);
    //}

    //printf("\naverage Host to Device Transfer Time: %f ms\n", total_H2D / 6);
    //printf("average Device to Host Transfer Time: %f ms\n", total_D2H / 6);




}

    //matrixMultiplicationKernel<<<threadsPerBlock, blocksPerGrid>>>(d_P, d_M, d_N, n); // sends 256 threads, and a gridsize
    //cudaMemcpy(a, d_a, nbytes, cudaMemcpyDeviceToHost, 0);
    //cudaEventRecord(stop, 0);

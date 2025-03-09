
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

#define TOLERANCE 0.0001
//TREVOR FARIAS 20321873

//NOTE: I wasn't sure how to include 3 seperate codes in one file so if you are testing, please uncomment the part you would like to test, and recomment the other parts

__host__ int retCores(cudaDeviceProp *device) {
    return 128 * device->multiProcessorCount; 
}

__global__ void matrixMultiplicationKernel(float *P, const float *M, const float *N, int numDimensions) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        for (int r = 0; r < numDimensions; r++) {
            for (int c = 0; c < numDimensions; c++) {
                float sum = 0.0f;
                for (int i = 0; i < numDimensions; i++) {
                    sum += M[r * numDimensions + i] * N[i * numDimensions + c];
                }
                P[r * numDimensions + c] = sum;
            }
        }
    }
}

__global__ void varyingMatrixMultiplicationKernel(float *P, float *M, float *N, int numDimensions) {
    //same approach as above roughly except it's spread over threads
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < numDimensions && c < numDimensions) {
        float sum = 0.0f;
        for (int k = 0; k < numDimensions; ++k) {
            sum += M[r * numDimensions + k] * N[k * numDimensions + c];
        }
        P[r * numDimensions + c] = sum;
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

    
    //part 2.1 - data collection  /////////////////////////////////////////////////////////////////////////////////////////////////
   /*
    int nd;
    float total_H2D = 0.0f, total_D2H = 0.0f;
    int n = 256; // 512 ctrl shift b -> ctrl f5

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

    //init the matrix's
    initializeMatrix(h_M, n * n);
    initializeMatrix(h_N, n * n);

    for (int i = 0; i < 6; i++) {
        float gpu_time = 0.0f;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaDeviceSynchronize();

        // host to Device Transfer Time
        cudaEventRecord(start, 0);
        cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_time, start, stop);
        total_H2D += gpu_time;
        printf(" %d - Host to Device Transfer Time: %f ms\n", i+1, gpu_time);

        // device to Host Transfer Time
        cudaEventRecord(start, 0);
        cudaMemcpy(h_M, d_M, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_N, d_N, size, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_time, start, stop);
        total_D2H += gpu_time;
        printf(" %d - Device to Host Transfer Time: %f ms\n", i+1, gpu_time);
    }
    printf("\naverage Host to Device Transfer Time: %f ms\n", total_H2D / 6);
    printf("average Device to Host Transfer Time: %f ms\n", total_D2H / 6);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    free(h_M);
    free(h_N);
    free(h_P);
    free(h_P_cpu);
    
    */

    //part 2.2 matrix mult /////////////////////////////////////////////////////////////////////////////////////////////////
    /*int n = 256; // 512 ctrl shift b -> ctrl f5

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

    //init the matrix's
    initializeMatrix(h_M, n * n);
    initializeMatrix(h_N, n * n);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();

    dim3 threadsPerBlock(1, 1);
    dim3 blocksPerGrid(1, 1);

    float gpu_time = 0.0f;
    float cpu_time = 0.0f;

    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
    cudaEventRecord(start, 0);
    matrixMultiplicationKernel <<<blocksPerGrid, threadsPerBlock>>> (d_P, d_M, d_N, n);
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
    */

    // part 2.3 Matrix Mult Varying Blocks /////////////////////////////////////////////////////////////////////////////////////////////////
    float cpu_time = 0.0f;
    for (int i = 0; i < 5; i++) {
        int x = pow(2, i + 8); //starts at 256
        size_t size = x * x * sizeof(float);
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

        //init the matrix's
        initializeMatrix(h_M, x * x);
        initializeMatrix(h_N, x * x);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaDeviceSynchronize();
        cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);


        for (int count = 1; count < 6; count++){
            int block = pow(2, count);

            dim3 threadsPerBlock(block, block);
            dim3 blocksPerGrid((x + block - 1) / block, (x + block - 1) / block);
            
            float gpu_time = 0.0f;
            cudaEventRecord(start, 0);
            varyingMatrixMultiplicationKernel<<<blocksPerGrid, threadsPerBlock>>>(d_P, d_M, d_N, x);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&gpu_time, start, stop);
            cudaMemcpy(h_P_cpu, d_P, size, cudaMemcpyDeviceToHost);
            printf("Matrix: %d x %d, Block: %d x %d, GPU Time: %f ms\n", x, x, block, block, gpu_time);
        }
        cudaEventRecord(start, 0);
        matrixMultiplicationCpu(h_P, h_M, h_N, x);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&cpu_time, start, stop);
        printf("CPU Time: %f ms\n", cpu_time);
        verifyMatrix(h_P, h_P_cpu, x);
       
        cudaFree(d_M);
        cudaFree(d_N);
        cudaFree(d_P);
        free(h_M);
        free(h_N);
        free(h_P);
        free(h_P_cpu);
    }
  
    
}
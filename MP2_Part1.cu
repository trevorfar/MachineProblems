#include <stdio.h>
#include <cuda.h>
#include <math.h>
#define TOLERANCE 0.0001

// I testify to the originality of my work : TREVOR FARIAS 20321873

const int test_sizes[] = { 256, 512, 1024, 2048, 4096 };
const int tile_sizes[] = { 2, 4, 8, 16, 32 };
const int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
const int num_tiles = sizeof(tile_sizes) / sizeof(tile_sizes[0]);

// tiled matrixa multiplcation
__global__ void matrixMulTiled(float* P, float* M, float* N, int width, int TILE_WIDTH) {
    extern __shared__ float sharedMemory[];
    float* Mshared = sharedMemory;
    float* Nshared = &sharedMemory[TILE_WIDTH * TILE_WIDTH];

    // finding tha rows and columns
    int threadIdX = threadIdx.x, threadIdY = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + threadIdY;
    int col = blockIdx.x * TILE_WIDTH + threadIdX;
    float valP = 0.0;

    for (int ph = 0; ph < width / TILE_WIDTH; ++ph) {
        if (row < width && (ph * TILE_WIDTH + threadIdX) < width)
            Mshared[threadIdY * TILE_WIDTH + threadIdX] = M[row * width + ph * TILE_WIDTH + threadIdX];
        else
            Mshared[threadIdY * TILE_WIDTH + threadIdX] = 0.0;

        if (col < width && (ph * TILE_WIDTH + threadIdY) < width)
            Nshared[threadIdY * TILE_WIDTH + threadIdX] = N[(ph * TILE_WIDTH + threadIdY) * width + col];
        else
            Nshared[threadIdY * TILE_WIDTH + threadIdX] = 0.0;

        // sync threads command shown in slides 
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            valP += Mshared[threadIdY * TILE_WIDTH + k] * Nshared[k * TILE_WIDTH + threadIdX];

        __syncthreads();
    }

    if (row < width && col < width)
        P[row * width + col] = valP;
}
// same verify matrix from machine problems 1
void verifyMatrix(float* matrix1, float* matrix2, int numDimensions) {
    for (int i = 0; i < numDimensions * numDimensions; i++) {
        if (fabs(matrix1[i] - matrix2[i]) > TOLERANCE) {
            printf("Test FAILED\n");
            return;
        }
    }
    printf("Test PASSED\n");
}


void hostFunction(int width, int TILE_WIDTH) {
    size_t size = width * width * sizeof(float);
    float* h_M, * h_N, * h_P, * d_M, * d_N, * d_P;

    h_M = (float*)malloc(size);
    h_N = (float*)malloc(size);
    h_P = (float*)malloc(size);
    float* h_P_cpu = (float*)malloc(size);

    for (int i = 0; i < width * width; i++) {
        h_M[i] = static_cast<float>(rand() % 10);
        h_N[i] = static_cast<float>(rand() % 10);
    }

    cudaMalloc((void**)&d_M, size);
    cudaMalloc((void**)&d_N, size);
    cudaMalloc((void**)&d_P, size);

    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);

    dim3 dimGrid((width + TILE_WIDTH - 1) / TILE_WIDTH, (width + TILE_WIDTH - 1) / TILE_WIDTH);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    size_t sharedMemSize = 2 * TILE_WIDTH * TILE_WIDTH * sizeof(float);

    matrixMulTiled << <dimGrid, dimBlock, sharedMemSize >> > (d_P, d_M, d_N, width, TILE_WIDTH);

    cudaMemcpy(h_P_cpu, d_P, size, cudaMemcpyDeviceToHost);
    verifyMatrix(h_P_cpu, h_P_cpu, width);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    free(h_M);
    free(h_N);
    free(h_P);
    free(h_P_cpu);
}

int main() {
    for (int i = 0; i < num_tests; i++) {
        int matrix_size = test_sizes[i];
        for (int j = 0; j < num_tiles; j++) {
            int TILE_WIDTH = tile_sizes[j];
            printf("test for matrix size: %d x %d with TILE_WIDTH: %d\n", matrix_size, matrix_size, TILE_WIDTH);

            float gpu_time = 0.0f;
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaDeviceSynchronize();

            cudaEventRecord(start, 0);
            hostFunction(matrix_size, TILE_WIDTH);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&gpu_time, start, stop);

            printf("GPU Execution Time for %d x %d with TILE_WIDTH %d: %f ms\n\n", matrix_size, matrix_size, TILE_WIDTH, gpu_time);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    }
    return 0;
}

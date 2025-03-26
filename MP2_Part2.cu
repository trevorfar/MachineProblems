#include <stdio.h>
#include <cuda.h>
#include <math.h>
#define TOLERANCE 0.0001

// I testify to the originality of my work : TREVOR FARIAS 20321873

const int TILE_WIDTH_X = 12;
const int TILE_WIDTH_Y = 18;

__global__ void matrixMulTiled(float* P, float* M, float* N, int M_rows, int M_cols, int N_cols) {
    extern __shared__ float sharedMemory[];
    float* Mshared = sharedMemory;
    float* Nshared = &sharedMemory[TILE_WIDTH_Y * TILE_WIDTH_X];

    int threadIdX = threadIdx.x, threadIdY = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH_Y + threadIdY;
    int col = blockIdx.x * TILE_WIDTH_X + threadIdX;
    float valP = 0.0;

    for (int ph = 0; ph < (M_cols + TILE_WIDTH_X - 1) / TILE_WIDTH_X; ++ph) {
        if (row < M_rows && (ph * TILE_WIDTH_X + threadIdX) < M_cols)
            Mshared[threadIdY * TILE_WIDTH_X + threadIdX] = M[row * M_cols + ph * TILE_WIDTH_X + threadIdX];
        else
            Mshared[threadIdY * TILE_WIDTH_X + threadIdX] = 0.0;

        if (col < N_cols && (ph * TILE_WIDTH_X + threadIdY) < M_cols)
            Nshared[threadIdY * TILE_WIDTH_X + threadIdX] = N[(ph * TILE_WIDTH_X + threadIdY) * N_cols + col];
        else
            Nshared[threadIdY * TILE_WIDTH_X + threadIdX] = 0.0;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH_X; ++k)
            valP += Mshared[threadIdY * TILE_WIDTH_X + k] * Nshared[k * TILE_WIDTH_X + threadIdX];

        __syncthreads();
    }

    if (row < M_rows && col < N_cols)
        P[row * N_cols + col] = valP;
}

void hostFunction(int M_rows, int M_cols, int N_cols) {
    size_t size_M = M_rows * M_cols * sizeof(float);
    size_t size_N = M_cols * N_cols * sizeof(float);
    size_t size_P = M_rows * N_cols * sizeof(float);

    float* h_M = (float*)malloc(size_M);
    float* h_N = (float*)malloc(size_N);
    float* h_P = (float*)malloc(size_P);
    float* d_M, * d_N, * d_P;

    for (int i = 0; i < M_rows * M_cols; i++) h_M[i] = static_cast<float>(rand() % 10);
    for (int i = 0; i < M_cols * N_cols; i++) h_N[i] = static_cast<float>(rand() % 10);

    cudaMalloc((void**)&d_M, size_M);
    cudaMalloc((void**)&d_N, size_N);
    cudaMalloc((void**)&d_P, size_P);

    cudaMemcpy(d_M, h_M, size_M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size_N, cudaMemcpyHostToDevice);

    dim3 dimGrid((N_cols + TILE_WIDTH_X - 1) / TILE_WIDTH_X, (M_rows + TILE_WIDTH_Y - 1) / TILE_WIDTH_Y);
    dim3 dimBlock(TILE_WIDTH_X, TILE_WIDTH_Y);
    size_t sharedMemSize = 2 * TILE_WIDTH_X * TILE_WIDTH_Y * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();

    cudaEventRecord(start, 0);
    matrixMulTiled << <dimGrid, dimBlock, sharedMemSize >> > (d_P, d_M, d_N, M_rows, M_cols, N_cols);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("GPU Execution Time for %d x %d * %d x %d with TILE_WIDTH (%d x %d): %f ms\n", M_rows, M_cols, M_cols, N_cols, TILE_WIDTH_X, TILE_WIDTH_Y, gpu_time);

    cudaMemcpy(h_P, d_P, size_P, cudaMemcpyDeviceToHost);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    free(h_M);
    free(h_N);
    free(h_P);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    printf("Running bonus test cases...\n");
    hostFunction(750, 800, 850);
    hostFunction(2000, 1750, 1900);
    return 0;
}

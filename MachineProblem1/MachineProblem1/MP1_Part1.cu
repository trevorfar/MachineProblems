
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

//TREVOR FARIAS 20321873

__host__ int retCores(cudaDeviceProp *device) {
    return 128 * device->multiProcessorCount; 
}


int main() {
    //part1 
    // device : NVIDIA 3060ti. device major = 8, according to online documentation this is an ampere convention which means 128 * the mp count. 
    int nd;
    cudaGetDeviceCount(&nd);
    for (int i = 0; i < nd; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        int cores = retCores(&deviceProps);
        printf("Number of devices: %d\n CUDA device [%s]\n Clock Rate: %d \n MultiProcessorCount: %d\n Number of Cores: %d\n", 
            nd, deviceProps.name, deviceProps.clockRate, deviceProps.multiProcessorCount, cores);
        printf(" warp size: %d\n total global memory: %zu \n shared mem per block %zu \n num registers per block %d \n max threads per block %d \n max dimension size of each block (%d, %d, %d) \n max size of grid (%d, %d, %d) \n",
            deviceProps.warpSize, deviceProps.totalGlobalMem, deviceProps.sharedMemPerBlock, deviceProps.regsPerBlock, deviceProps.maxThreadsPerBlock, deviceProps.maxThreadsDim[0],
            deviceProps.maxThreadsDim[1], deviceProps.maxThreadsDim[2], deviceProps.maxGridSize[0], deviceProps.maxGridSize[1], deviceProps.maxGridSize[2]);
    }
   
}

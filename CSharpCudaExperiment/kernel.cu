#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
#include <stdio.h>

extern "C" {
    __global__ void FactorKernel(int* m, int v)
    {
        //int i = threadIdx.x + (blockDim.x * blockIdx.x);
        //m[i] = v*i;
        m[threadIdx.x + (blockDim.x * blockIdx.x)] *= v;
    }

    __global__ void SetKernel(int* m, int v)
    {
        m[threadIdx.x + (blockDim.x * blockIdx.x)] = v;
    }

    __global__ void AddKernel(int* m, int v)
    {
        m[threadIdx.x + (blockDim.x * blockIdx.x)] += v;
    }

    __global__ void GetEnergy(float* x, float* y, float* z, int i)
    {
        //http://cuda-programming.blogspot.com/2013/01/vector-dot-product-in-cuda-c-cuda-c.html
    }

    __global__ void SequenceProduct (const int N, const float* V1, const float* V2, float* V3)
    {
        const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < N)
            V3[tid] = V1[tid] * V2[tid];
    }

    __global__ void VectorSum(const int N, const float* v, float * sum)
    {
        __shared__ float chache[1024];
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        const unsigned int chacheindex = threadIdx.x;

        float temp = 0;
        while (tid < N)
        {
            temp += v[tid];
            tid += blockDim.x * gridDim.x;
        }
        chache[chacheindex] = temp;
        __syncthreads();

        int i = blockDim.x / 2;
        while (i != 0)
        {
            if (chacheindex < i)
                chache[chacheindex] += chache[chacheindex + i];
            __syncthreads();
            i /= 2;
        }
        if (chacheindex == 0)
            sum[blockIdx.x] = chache[0];
    }

    __global__ void VectorDotProduct (const int N, const float* V1, const float* V2, float* V3)
    {
        __shared__ float chache[1024];
        float temp;
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        const unsigned int chacheindex = threadIdx.x;

        while (tid < N)
        {
            temp += V1[tid] * V2[tid];
            tid += blockDim.x * gridDim.x;
        }
        chache[chacheindex] = temp;
        __syncthreads();

        int i = blockDim.x / 2;
        while (i != 0)
        {
            if (chacheindex < i)
                chache[chacheindex] += chache[chacheindex + i];
            __syncthreads();
            i /= 2;
        }
        if (chacheindex == 0)
            V3[blockIdx.x] = chache[0];
    }
}

// Print device properties
void printDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n", devProp.major);
    printf("Minor revision number:         %d\n", devProp.minor);
    printf("Name:                          %s\n", devProp.name);
    printf("Total global memory:           %u\n", devProp.totalGlobalMem);
    printf("Total shared memory per block: %u\n", devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n", devProp.regsPerBlock);
    printf("Warp size:                     %d\n", devProp.warpSize);
    printf("Maximum memory pitch:          %u\n", devProp.memPitch);
    printf("Maximum threads per block:     %d\n", devProp.maxThreadsPerBlock);
    for (int i = 1; i <= 3; ++i)
        printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i - 1]);
    for (int i = 1; i <= 3; ++i)
        printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i - 1]);
    printf("Clock rate:                    %d\n", devProp.clockRate);
    printf("Total constant memory:         %u\n", devProp.totalConstMem);
    printf("Texture alignment:             %u\n", devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n", (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n", (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    return;
}

int main()
{
    // Number of CUDA devices
    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("CUDA Device Query...\n");
    printf("There are %d CUDA devices.\n", devCount);

    // Iterate through devices
    for (int i = 0; i < devCount; ++i)
    {
        // Get device properties
        printf("\nCUDA Device #%d\n", i);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printDevProp(devProp);
    }

    printf("\nPress any key to exit...");
    char c;
    scanf("%c", &c);

    return 0;
}
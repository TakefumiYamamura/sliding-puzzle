#include <cuda_runtime.h>
#include <stdio.h>

int
main(int argc, char **argv)
{
    int         devcnt = 0;
    cudaError_t e      = cudaGetDeviceCount(&devcnt);

    if (e != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n%s\n", (int) e,
               cudaGetErrorString(e));
        exit(EXIT_FAILURE);
    }

    printf("%d available CUDA devices\n", devcnt);

    {
        int dev = 0, driverVersion = 0, runtimeVersion = 0;
        cudaSetDevice(dev);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        printf("Device %d: %s\n", dev, prop.name);

        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("CUDA driver version / runtime version %d.%d / %d.%d\n",
               driverVersion / 1000, driverVersion % 100 / 10,
               runtimeVersion / 1000, runtimeVersion % 100 / 10);
        printf("CUDA compute capability: %d.%d\n", prop.major, prop.minor);
        printf("Total amount of global memory: %.2f MBytes = %llu bytes\n",
               (float) prop.totalGlobalMem / (pow(1024.0, 2)),
               (unsigned long long) prop.totalGlobalMem);
        printf("GPU clock rate: %.0f MHz = %0.2f GHz\n", prop.clockRate * 1e-3f,
               prop.clockRate * 1e-6f);
        printf("Memory clock rate: %.0f MHz\n", prop.memoryClockRate * 1e-3f);
        printf("Memory bus width: %d-bit\n", prop.memoryBusWidth);
        if (prop.l2CacheSize)
            printf("L2 cache size: %d bytes\n", prop.l2CacheSize);
        printf("Max texture Dimesnios size(x,y,z) 1D=(%d), 2D=(%d,%d), "
               "3D=(%d,%d,%d)\n",
               prop.maxTexture1D, prop.maxTexture2D[0], prop.maxTexture2D[1],
               prop.maxTexture3D[0], prop.maxTexture3D[1],
               prop.maxTexture3D[2]);
        printf("max layered texture size (dim) * layers"
               "1D=(%d) * %d, 2D=(%d,%d) & %d\n",
               prop.maxTexture1DLayered[0], prop.maxTexture1DLayered[1],
               prop.maxTexture2DLayered[0], prop.maxTexture2DLayered[1],
               prop.maxTexture2DLayered[2]);
        printf("Total amount of constant memory: %lu bytes\n",
               prop.totalConstMem);
        printf("Total amount of shared memory per block: %lu bytes\n",
               prop.sharedMemPerBlock);
        printf("Total number of registers available per block: %d\n",
               prop.regsPerBlock);
        printf("Warp size: %d\n", prop.warpSize);
        printf("Maximum number of threads per multiprocessor: %d\n",
               prop.maxThreadsPerMultiProcessor);
        printf("Maximum number of threads per block: %d\n",
               prop.maxThreadsPerBlock);
        printf("Maximum sizes of each dimension of a block: %d * %d * %d\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1],
               prop.maxThreadsDim[2]);
        printf("Maximum sizes of each dimension of a grid: %d * %d * %d\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("Maximum memory pitch: %lu bytes\n", prop.memPitch);
    }
    exit(EXIT_SUCCESS);
}

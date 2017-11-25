#include <stdio.h>
#ifndef UNABLE_LOG
#define elog(...) fprintf(stderr, __VA_ARGS__)
#else
#define elog(...) ;
#endif

#define exit_failure(...)                                                      \
    do                                                                         \
    {                                                                          \
        printf(__VA_ARGS__);                                                   \
        exit(EXIT_FAILURE);                                                    \
    } while (0)
#define CUDA_CHECK(call)                                                       \
    do                                                                         \
    {                                                                          \
        const cudaError_t e = call;                                            \
        if (e != cudaSuccess)                                                  \
            exit_failure("Error: %s:%d code:%d, reason: %s\n", __FILE__,       \
                         __LINE__, e, cudaGetErrorString(e));                  \
    } while (0)

__global__ void kernel(int *ptr, int *plan)
{
	plan[threadIdx.x] = ptr[threadIdx.x] * 10;
}

__host__ static void *
cudaPalloc(size_t size)
{
    void *ptr;
    elog("palloc size=%zu\n", size);
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

__host__ static void
cudaPfree(void *ptr)
{
    CUDA_CHECK(cudaFree(ptr));
}

#define SIZE (n * sizeof(int))
int main(void)
{
	int n = 16;
	int ar[123456];
	int plan[124356];
	for (int i = 0; i < 123456; ++i)
		ar[i] = i;
	int * d_ptr = (int *) cudaPalloc(SIZE);
	int * d_plan = (int *) cudaPalloc(SIZE);
	for (int i = 0; i < 10; ++i)
	{
		CUDA_CHECK(cudaMemset(d_ptr, 0,  SIZE));
		CUDA_CHECK(cudaMemcpy(d_ptr, ar, SIZE, cudaMemcpyHostToDevice));
		kernel<<<1, 1>>>(d_ptr, d_plan);
		CUDA_CHECK(cudaMemcpy(plan, d_plan, SIZE, cudaMemcpyDeviceToHost));
		n *= 2;
		cudaPfree(d_ptr);
		d_ptr = (int *) cudaPalloc(SIZE);
		cudaPfree(d_plan);
		d_plan = (int *) cudaPalloc(SIZE);
	}
	cudaPfree(d_ptr);
	cudaPfree(d_plan);
	return 0;
}

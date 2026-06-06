// softmax function

#include "softmax.cuh"
#include <cuda_runtime.h>

__global__ void softmax_kernel(float*x, int cols){
    extern __shared__ float sdata[]; // declares a shared meemory array whose size is NOT known at compile time.
    int row = blockIdx.x;
    float* row_ptr = x + row * cols; // pointer to the start of the current block's row.

    float thread_max = -1e38f;// any real val will be larger than this
    for(int i = threadIdx.x; i < cols; i += blockDim.x){
        thread_max = fmaxf(thread_max, row_ptr[i]);
    }
}
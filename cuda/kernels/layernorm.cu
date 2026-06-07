#include "layernorm.cuh"
#include <cuda_runtime.h>

__global__ void layernorm_fwd_kernel(
    float* out,
    const float* in,
    const float* gamma,
    const float* beta,
    int d_model, 
    float eps) {
    extern __shared__ float sdata[];

    int token = blockIdx.x;
    const float* x = in + token * d_model;
    float* y = out + token * d_model;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        sum += x[i];
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1){

        if (threadIdx.x < s){
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
            __syncthreads();
        }
        
    }
    float mean = sdata[0] / d_model;
}
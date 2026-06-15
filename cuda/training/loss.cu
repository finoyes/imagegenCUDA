#include "loss.cuh"
#include <cuda_runtime.h>

__global__ void mse_grad_kernel(float* grad, const float* pred, const float* target, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad[i] = 2.0f * (pred[i] - target[i]) / n;
}

__global__ void mse_loss_kernel(float* loss, const float* pred, const float* target, int n){
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;

    float val = 0.0f;
    if (i < n) {
        float diff = pred[i] - target[i];
        val = diff * diff;
    }
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(loss, sdata[0] / n);  // accumulate from all blocks
}//wirte

void mse_loss(float* loss_out, float* grad_out,
              const float* pred, const float* target, int n)
{
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;

    // Reset loss accumulator
    cudaMemset(loss_out, 0, sizeof(float));

    mse_loss_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        loss_out, pred, target, n);
    mse_grad_kernel<<<blocks, threads>>>(
        grad_out, pred, target, n);
}// write
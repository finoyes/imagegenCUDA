#include "matmul.cuh"
#include <cuda_runtime.h>

#define TILE 16

__global__ void matrixmult_kernel(

    const float* __restrict__ A, // A and B don't misidentify C(overlap (dont point to overlapping memory))
    const float* __restrict__ B,
    float* C,
    int M,
    int K,
    int N,
    float alpha,
    float beta

    // parameters M,N,K, alpha, beta are passed as scalars- in constant registers on the GPU, avail to every thread instantly.

);

{
    __shared__ float As[TILE][TILE]; // shared memory for tiles of A and B, faster than global memory.
    __shared__ float Bs[TILE][TILE];
    
    int row = blockId.y * TILE + threadIdx.y;
    int col = blockId.x * TILE + threadIdx.x;

    float acc = 0.0f; // dot prod accumulator.
 
    for (int t = 0; t < (K+ TILE - 1)/TILE; t++){

        if(row < M && t * TILE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t *T threadIdx.x];
       else
            As[threadIdx.y][threadIdx.x] = 0.0f;     

    }
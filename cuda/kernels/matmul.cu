//matrix multiplication which is used everywhere in transformer.

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

)
{
    __shared__ float As[TILE][TILE]; // shared memory for tiles of A and B, faster than global memory.
    __shared__ float Bs[TILE][TILE];
    
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f; // dot prod accumulator.
 
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {

        if(row < M && t * TILE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;     

        if(col < N && t * TILE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;   
            
        __syncthreads(); // ensure all threads have loaded their tile before computation.

        for(int k = 0; k < TILE; k++) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x]; //  actual computation for TILE iterations
        }

        __syncthreads();
    }

    if(row < M && col < N) {
      if(beta == 0.0f)
        C[row * N + col] = alpha * acc; // if beta is 0, we can skip reading C and just write the result.
      else
        C[row * N + col] = alpha * acc + beta * C[row * N + col]; // otherwise, we need to read C, scale it by beta, and add it to the new value.
    }
}
//processing all the tiles write to C.

void matrixmult(const float* A, const float* B, float* C, int M, int K, int N, float alpha, float beta){

    dim3 block(TILE, TILE);// dim3 = 3D dimentions in CUDA. z is defaulted to 1 as unused.
    dim3 grid((N + TILE - 1) / TILE, (M + TILE -1) / TILE);// how many blocks to launch.

    matrixmult_kernel<<<grid, block>>>(A, B, C, M, K, N, alpha, beta);// CUDA kernel launch , passess the execution config in <<< >>> = how many blocks and threads to launch.


}// CPU launcher function

__global__ void batched_matrixmult_kernel(const float* A, const float* B, float* C, int M, int K, int N){
 
    int b = blockIdx.z; // 3rd dimension of grid for batched matmult.
    const float* Ab = A + b * M * K; // pointer to the b-th matrix in A batch.
    const float* Bb = B + b * K * N; // pointer to the b-th matrix in B batch.
    float* Cb = C + b * M * N; // pointer to the b-th matrix in C batch.

  
}

void batched_matrixmult(const float* A, const float* B, float* C, int M, int K, int N, int batch){
   
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE -1) / TILE, batch); // 3rd dimension for batch size.

    batched_matrixmult_kernel<<<grid, block>>>(A, B, C, M, K, N);

}

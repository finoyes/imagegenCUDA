//matrix multiplication which is used everywhere in transformer.

#include "matmul.cuh"
#include <cuda_runtime.h>

#define TILE 16

// ── Standard tiled GEMM: C = alpha*A*B + beta*C ─────────────────────────────
// A: M×K, B: K×N, C: M×N
__global__ void matrixmult_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* C,
    int M, int K, int N,
    float alpha, float beta)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        As[threadIdx.y][threadIdx.x] = (row < M && t*TILE+threadIdx.x < K) ?
            A[row*K + t*TILE + threadIdx.x] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (col < N && t*TILE+threadIdx.y < K) ?
            B[(t*TILE+threadIdx.y)*N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; k++) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) {
        if (beta == 0.0f)
            C[row*N + col] = alpha * acc;
        else
            C[row*N + col] = alpha * acc + beta * C[row*N + col];
    }
}

void matrixmult(const float* A, const float* B, float* C, int M, int K, int N,
                float alpha, float beta)
{
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matrixmult_kernel<<<grid, block>>>(A, B, C, M, K, N, alpha, beta);
}

// ── C(K×N) += A^T * B  (A stored as M×K, B stored as M×N) ──────────────────
// Used for weight-gradient accumulation: dW += X^T * dY
__global__ void matmul_AtB_kernel(const float* A, const float* B, float* C,
                                   int M, int K, int N)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    // row of C → column of A (k dimension), col of C → col of B (n dimension)
    int row = blockIdx.y * TILE + threadIdx.y;  // k index
    int col = blockIdx.x * TILE + threadIdx.x;  // n index
    float acc = 0.0f;

    for (int t = 0; t < (M + TILE - 1) / TILE; t++) {
        // A^T[row, t*TILE+threadIdx.x] = A[t*TILE+threadIdx.x, row]
        int m_a = t * TILE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (m_a < M && row < K) ? A[m_a*K + row] : 0.0f;
        // B[t*TILE+threadIdx.y, col]
        int m_b = t * TILE + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (m_b < M && col < N) ? B[m_b*N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; k++) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }

    if (row < K && col < N) C[row*N + col] += acc;  // += for gradient accumulation
}

void matmul_AtB(const float* A, const float* B, float* C, int M, int K, int N)
{
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (K + TILE - 1) / TILE);
    matmul_AtB_kernel<<<grid, block>>>(A, B, C, M, K, N);
}

// ── C(M×N) = A * B^T  (A stored as M×K, B stored as N×K) ───────────────────
// Used for input-gradient computation: dX = dY * W^T
__global__ void matmul_ABt_kernel(const float* A, const float* B, float* C,
                                   int M, int K, int N)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;  // m index
    int col = blockIdx.x * TILE + threadIdx.x;  // n index
    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        // A[row, t*TILE+threadIdx.x]
        int k_a = t * TILE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < M && k_a < K) ? A[row*K + k_a] : 0.0f;
        // B^T[t*TILE+threadIdx.y, col] = B[col, t*TILE+threadIdx.y]
        int k_b = t * TILE + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (col < N && k_b < K) ? B[col*K + k_b] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; k++) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) C[row*N + col] = acc;  // overwrites
}

void matmul_ABt(const float* A, const float* B, float* C, int M, int K, int N)
{
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_ABt_kernel<<<grid, block>>>(A, B, C, M, K, N);
}

// ── Batched GEMM: C[b] = A[b] * B[b]  ───────────────────────────────────────
// A: (batch × M × K), B: (batch × K × N), C: (batch × M × N)
__global__ void batched_matrixmult_kernel(const float* A, const float* B, float* C,
                                           int M, int K, int N)
{
    int b = blockIdx.z;
    const float* Ab = A + (size_t)b * M * K;
    const float* Bb = B + (size_t)b * K * N;
    float*       Cb = C + (size_t)b * M * N;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        As[threadIdx.y][threadIdx.x] = (row < M && t*TILE+threadIdx.x < K) ?
            Ab[row*K + t*TILE + threadIdx.x] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (col < N && t*TILE+threadIdx.y < K) ?
            Bb[(t*TILE+threadIdx.y)*N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; k++) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) Cb[row*N + col] = acc;
}

void batched_matrixmult(const float* A, const float* B, float* C,
                        int batch, int M, int K, int N)
{
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, batch);
    batched_matrixmult_kernel<<<grid, block>>>(A, B, C, M, K, N);
}

// ── Batched C[b](K×N) += A[b]^T * B[b]  ─────────────────────────────────────
// A: (batch × M × K), B: (batch × M × N), C: (batch × K × N), accumulated
__global__ void batched_matmul_AtB_kernel(const float* A, const float* B, float* C,
                                           int M, int K, int N)
{
    int b = blockIdx.z;
    const float* Ab = A + (size_t)b * M * K;
    const float* Bb = B + (size_t)b * M * N;
    float*       Cb = C + (size_t)b * K * N;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;  // k index
    int col = blockIdx.x * TILE + threadIdx.x;  // n index
    float acc = 0.0f;

    for (int t = 0; t < (M + TILE - 1) / TILE; t++) {
        int m_a = t * TILE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (m_a < M && row < K) ? Ab[m_a*K + row] : 0.0f;
        int m_b = t * TILE + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (m_b < M && col < N) ? Bb[m_b*N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; k++) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }

    if (row < K && col < N) Cb[row*N + col] += acc;
}

void batched_matmul_AtB(const float* A, const float* B, float* C,
                        int batch, int M, int K, int N)
{
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (K + TILE - 1) / TILE, batch);
    batched_matmul_AtB_kernel<<<grid, block>>>(A, B, C, M, K, N);
}

// ── Batched C[b](M×N) = A[b] * B[b]^T  ──────────────────────────────────────
// A: (batch × M × K), B: (batch × N × K), C: (batch × M × N), overwrite
__global__ void batched_matmul_ABt_kernel(const float* A, const float* B, float* C,
                                           int M, int K, int N)
{
    int b = blockIdx.z;
    const float* Ab = A + (size_t)b * M * K;
    const float* Bb = B + (size_t)b * N * K;
    float*       Cb = C + (size_t)b * M * N;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;  // m index
    int col = blockIdx.x * TILE + threadIdx.x;  // n index
    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int k_a = t * TILE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < M && k_a < K) ? Ab[row*K + k_a] : 0.0f;
        int k_b = t * TILE + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (col < N && k_b < K) ? Bb[col*K + k_b] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; k++) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) Cb[row*N + col] = acc;
}

void batched_matmul_ABt(const float* A, const float* B, float* C,
                        int batch, int M, int K, int N)
{
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, batch);
    batched_matmul_ABt_kernel<<<grid, block>>>(A, B, C, M, K, N);
}

// ── Element-wise accumulation: dst += src ────────────────────────────────────
__global__ void vec_add_inplace_kernel(float* dst, const float* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

void vec_add_inplace(float* dst, const float* src, int n) {
    vec_add_inplace_kernel<<<(n + 255) / 256, 256>>>(dst, src, n);
}

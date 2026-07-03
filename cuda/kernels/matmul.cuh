#pragma once
#include <cuda_runtime.h>

// Standard GEMM: C = alpha*A*B + beta*C  (A:M×K, B:K×N, C:M×N)
void matrixmult(const float* A, const float* B, float* C,
                int M, int K, int N,
                float alpha = 1.0f, float beta = 0.0f);

// C(K×N) += A^T * B  (A stored as M×K, B stored as M×N)
// Used for weight-gradient accumulation: dW += X^T * dY
void matmul_AtB(const float* A, const float* B, float* C, int M, int K, int N);

// C(M×N) = A * B^T  (A stored as M×K, B stored as N×K)
// Used for input-gradient computation: dX = dY * W^T
void matmul_ABt(const float* A, const float* B, float* C, int M, int K, int N);

// Batched GEMM: C[b] = A[b]*B[b]  (A:batch×M×K, B:batch×K×N, C:batch×M×N)
void batched_matrixmult(const float* A, const float* B, float* C,
                        int batch, int M, int K, int N);

// Batched C[b](K×N) += A[b]^T * B[b]  (A:batch×M×K, B:batch×M×N, C:batch×K×N)
void batched_matmul_AtB(const float* A, const float* B, float* C,
                        int batch, int M, int K, int N);

// Batched C[b](M×N) = A[b] * B[b]^T  (A:batch×M×K, B:batch×N×K, C:batch×M×N)
void batched_matmul_ABt(const float* A, const float* B, float* C,
                        int batch, int M, int K, int N);

// Element-wise accumulation: dst[i] += src[i]
void vec_add_inplace(float* dst, const float* src, int n);

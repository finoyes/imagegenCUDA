//matrix multiplication kernel for the transformer.

#pragma once
#include <cuda_runtime.h>

void matrixmult(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N,

    float alpha = 1.0f, /*enables BLAS signature: C = alpha * A * B + beta * C with defaults its just C = AB.
    beta = 1 would let you accumulate into C instead of overwrting it.*/
    float beta = 0.0f // CPU side floats passed to the kernel as scalars.
);


void batched_matrixmult(
    const float* A,
    const float* B,
    float* C,
    int batch, //batch size
    int M,
    int K,
    int N
    
); // same operation repeated batch times independently. also used in attentions (batch*heads) where indie matmults are happening simultaneously.


//activation funtion kernel (GeLU).
// GeLU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi * (x + 0.044715 * x*x*x)))
//not ReLU cuz solves dying ReLU problem. leaky ReLU also solves it but not  as effectively.

#pragma once
#include <cuda_runtime.h>

void GeLU_forward(float* output, const float* input, int n);// n = total element count( batchxseqxd_model).
void GeLU_backward(float* d_input, const float* d_output, const float* input, int n);// d_output(upstream gradient) and input(og forward, input to compute the derivative)
void ReLU_forward(float* output, const float* input, int n);
void ReLU_backward(float* d_input, const float* d_output, const float* input, int n);
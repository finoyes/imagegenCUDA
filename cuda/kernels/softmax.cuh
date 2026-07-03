//converts the raw attention scores to probabilities using the softmax function.

#pragma once
#include <cuda_runtime.h>

void softmax(float* input, int rows, int cols);//row-wise softwmax, input = 2D matrix -> rowxcol.
void attention_softmax(float* scores, int batch, int heads, int seq_len); //convinience wrapper
// Backward through row-wise softmax: dx[i] = s[i]*(ds[i] - dot(ds,s)) per row
void attention_softmax_backward(float* dx, const float* s, const float* ds,
                                int batch, int heads, int seq_len);

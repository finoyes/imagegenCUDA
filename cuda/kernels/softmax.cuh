//converts the raw attention scores to probabilities using the softmax(kernel).

#pragma once
#include <cuda_runtime.h>

void softmax(float* input, int rows, int cols);//row-wise softwmax, input = 2D matrix -> rowxcol.
void attention_softmax(float* scores, int batch, int heads, int seq_len); //convinience wrapper, attention score tensor is 4D (batch, heads, seq_len, seq_len) but we can treat the last two dimensions as a single 2D matrix for softmax. so we pass batch*heads as rows and seq_len*seq_len as cols.

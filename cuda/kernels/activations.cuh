//activation funtion kernel (GeLU).
// GeLU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x*x*x))
//not ReLU cuz solves dying ReLU problem. leaky ReLU also solves it but not  as effectively.

#pragma once
#include <cuda_runtime.h>

void GeLU_forward();
void GeLU_backward();
void ReLU_forward();
void ReLU_backward();
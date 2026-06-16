/*mean squared error loss and it's gradient.

L = (1/N) Σ (y_pred - y_true)²
dL/dy_pred = (2/N) * (y_pred - y_true)*/

#pragma once
#include <cuda_runtime.h>

void mse_loss(
    float* loss_out,
    float* grad_out,
    const float* pred,
    const float* target,
    int n
);

//explain the file

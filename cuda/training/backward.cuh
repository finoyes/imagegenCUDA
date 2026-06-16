/*Backpropagation is the chain rule applied recursively through the computation
graph. For each operation `y = f(x)`, backprop computes `dx = (dy/dx) * d_loss/dy`.
*/

#pragma once
#include <cuda_runtime.h>
#include "../model/transformer.cuh"
#include "optimizer.cuh"

// Run the full backward pass through num_layers transformer blocks
// d_loss_output: gradient of loss w.r.t. transformer output (from decoder backward)
// Fills in all dW, db fields in params and writes d_encoder_output
void transformer_backward_full(
    float* d_encoder_output,          // gradient to pass back to encoder
    TransformerParams* params,
    const float* d_loss_output,
    const TransformerCache* cache,
    int batch, int seq_len
);

// Gradient clipping — prevents exploding gradients
// Scales all gradients so their global L2 norm <= max_norm
void clip_grad_norm(AdamWOptimizer* opt, float max_norm);
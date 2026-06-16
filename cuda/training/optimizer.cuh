/*AdamW - adam with decoupled weigjt decay, it is the standard optimizer for transformers, maintains a running estimate of the gradient mean(m
and incentered varience(v) for each parameter, and uses these to pruduce adaptive per-parameter learning rates

g_t  = gradient at step t
m_t  = β₁ * m_{t-1} + (1 - β₁) * g_t           (first moment — gradient mean)
v_t  = β₂ * v_{t-1} + (1 - β₂) * g_t²          (second moment — gradient variance)
m̂_t  = m_t / (1 - β₁ᵗ)                         (bias correction)
v̂_t  = v_t / (1 - β₂ᵗ)                         (bias correction)
θ_t  = θ_{t-1} * (1 - lr * λ)  -  lr * m̂_t / (√v̂_t + ε)

The `(1 - lr * λ)` term is the **weight decay** applied directly to the
parameter — this is what makes it AdamW vs Adam (in plain Adam, weight decay
is mixed into the gradient, which interacts badly with the adaptive learning
rate scaling).

Standard values: `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`, `λ = 0.01`

*/

#pragma once
#include <cuda_runtime.h>
#include <vector>
using namespace std;

struct AdamWState {
    float* m;// first moment buffer
    float* v;// second moment buffer
    int n;// number of parameters in this tensor

};

struct AdamWOptimizer{

    vector<float*>     params;// device pointers to wieght tensors.
    vector<float*>     gradient;// device pointers to gradient tensors.
    vector<AdamWState> states;// m, v buffers per tensor
    vector<int>        sizes;// element count per tensor

    float lr; // learning rate
    float beta1; //0.9
    float beta2; //0.999
    float eps; //1e-8
    float weight_decay; //0.01
    int step; // current step( for bias correction)

};

AdamWOptimizer adamw_create(float lr, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, float weight_decay = 0.01f);

void adamw_add_param(AdamWOptimizer* opt, float* param, float* grad, int n);
void adamw_step(AdamWOptimizer* opt);
void adamw_zero_grad(AdamWOptimizer* opt);
void adamw_destroy(AdamWOptimizer* opt);
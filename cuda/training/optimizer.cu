#include "optimizer.cuh"
#include <cuda_runtime.h>
#include <math.h>

__global__ void adamw_updater_kernel(float* param, float* gradient, float* m, float* v, float lr, float beta1, float beta2, float eps, float wd, float bias_corr1, float bias_corr2, int n){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    float g = gradient[i];

    m[i] = beta1 * m[i] + (1.0f - beta1) * g;
    v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;//write

    float m_hat = m[i] / bias_corr1;
    float v_hat = v[i] / bias_corr2;

    param[i] = param[i] * (1.0f - lr * wd) - lr * m_hat / (sqrtf(v_hat) + eps);
//problem in the expression
}

AdamWOptimizer adamw_create(float lr, float beta1, float beta2, float eps, float weight_decay){

    AdamWOptimizer opt;
    opt.lr = lr; opt.beta1 = beta1; opt.beta2 = beta2;
    opt.eps = eps; opt.weight_decay = weight_decay;
    opt.step = 0;
    return opt;
}

void adamw_add_param(AdamWOptimizer* opt, float* param, float* grad, int n){

    AdamWState state;
    state.n = n;
    cudaMalloc(&state.m, n * sizeof(float));
    cudaMalloc(&state.v, n * sizeof(float));
    cudaMemset(state.m, 0, n * sizeof(float));
    cudaMemset(state.v, 0, n * sizeof(float));

    opt->params.push_back(param);
    opt->gradient.push_back(grad);
    opt->states.push_back(state);
    opt->sizes.push_back(n);
    
}

void adamw_step(AdamWOptimizer* opt){
     opt->step++;
     float bc1 = 1.0f - powf( opt->beta1, opt->step);
     float bc2 = 1.0f - powf( opt->beta2, opt->step);

     int threads = 256;
     for (int i = 0; i < (int)opt->params.size(); i++){
        int n = opt->sizes[i];
        adamw_updater_kernel<<<(n+threads-1)/threads, threads>>>( opt->params[i], opt->gradient[i], opt->states[i].m, opt->states[i].v, opt->lr, opt->beta1, opt->beta2, opt->eps, opt->weight_decay, bc1, bc2, n );
     }
}

void adamw_zero_grad(AdamWOptimizer* opt) {
    for (int i = 0; i < (int)opt->gradient.size(); i++)
        cudaMemset(opt->gradient[i], 0, opt->sizes[i] * sizeof(float));
}

void adamw_destroy(AdamWOptimizer* opt) {
    for (auto& s : opt->states) {
        cudaFree(s.m);
        cudaFree(s.v);
    }
}

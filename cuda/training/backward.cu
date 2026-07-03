#include "backward.cuh"
#include "../kernels/matmul.cuh"
#include <cuda_runtime.h>
#include <math.h>

// ── Gradient clipping ─────────────────────────────────────────────────────────

__global__ void l2_norm_kernel(const float* x, float* norm_sq, int n) {
    extern __shared__ float sdata[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[threadIdx.x] = (i < n) ? x[i] * x[i] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(norm_sq, sdata[0]);
}

__global__ void scale_grads_kernel(float* x, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= scale;
}

void clip_grad_norm(AdamWOptimizer* opt, float max_norm) {
    float* d_norm_sq;
    cudaMalloc(&d_norm_sq, sizeof(float));
    cudaMemset(d_norm_sq, 0, sizeof(float));

    for (int i = 0; i < (int)opt->gradient.size(); i++) {
        int n = opt->sizes[i];
        l2_norm_kernel<<<(n+255)/256, 256, 256*sizeof(float)>>>(
            opt->gradient[i], d_norm_sq, n);
    }

    float norm_sq;
    cudaMemcpy(&norm_sq, d_norm_sq, sizeof(float), cudaMemcpyDeviceToHost);
    float norm = sqrtf(norm_sq);
    cudaFree(d_norm_sq);

    if (norm > max_norm) {
        float scale = max_norm / norm;
        for (int i = 0; i < (int)opt->gradient.size(); i++) {
            int n = opt->sizes[i];
            scale_grads_kernel<<<(n+255)/256, 256>>>(opt->gradient[i], scale, n);
        }
    }
}

// ── transformer_backward_full ─────────────────────────────────────────────────
// Thin wrapper: delegates to transformer_backward (implemented in transformer.cu).

void transformer_backward_full(
    float* d_encoder_output,
    TransformerParams* params,
    const float* d_loss_output,
    const TransformerCache* cache,
    int batch, int seq_len)
{
    transformer_backward(d_encoder_output, params, d_loss_output, cache, batch, seq_len);
}
#include "backward.cuh"
#include "../kernels/matmul.cuh"
#include "../kernels/layernorm.cuh"
#include "../kernels/activations.cuh"
#include "optimizer.cuh"
#include <cuda_runtime.h>
#include <math.h>

// Compute L2 norm of a tensor (for gradient clipping)
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

/*__global__ void scale_gradient_kernel(float* grad, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad[idx] *= scale;
    }
}
*/
void clip_grad_norm(AdamWOptimizer* opt, float max_norm) {
    // 1. Compute global norm across all parameter gradients
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

    // 2. If norm exceeds max_norm, scale all gradients down
    if (norm > max_norm) {
        float scale = max_norm / norm;
        for (int i = 0; i < (int)opt->gradient.size(); i++) {
            int n = opt->sizes[i];
            scale_grads_kernel<<<(n+255)/256, 256>>>(opt->gradient[i], scale, n);//prblm
        }
    }
}

void transformer_backward_full(
    float* d_encoder_output, TransformerParams* params,
    const float* d_loss_output, const TransformerCache* cache,
    int batch, int seq_len)
{
    // Allocate a working gradient buffer (same size as one layer's activations)
    int N = batch * seq_len, D = params->d_model;
    float* d_current;
    cudaMalloc(&d_current, N * D * sizeof(float));
    cudaMemcpy(d_current, d_loss_output, N * D * sizeof(float), cudaMemcpyDeviceToDevice);

    for (int l = params->num_layers - 1; l >= 0; l--) {
        // Backprop through each sub-layer in reverse order
        // This is where you call layernorm_backward, attention_backward,
        // and the matmul gradient operations described above
        // Each step updates the dW/db fields in params->blocks[l]
        // and passes d_current backwards
    }

    cudaMemcpy(d_encoder_output, d_current, N * D * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(d_current);
}
#include "layernorm.cuh"
#include <cuda_runtime.h>

__global__ void layernorm_fwd_kernel(
    float* out,
    const float* in,
    const float* gamma,
    const float* beta,
    int d_model, 
    float eps) {
    extern __shared__ float sdata[];

    int token = blockIdx.x;
    const float* x = in + token * d_model;
    float* y = out + token * d_model;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        sum += x[i];
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1){

        if (threadIdx.x < s){
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    float mean = sdata[0] / d_model;

    float var_sum = 0.0f;
    for(int i = threadIdx.x;i < d_model; i += blockDim.x){
        float diff = x[i] - mean;
        var_sum += diff * diff; /* squared deviation from mean. We sum these and divide by
d_model to get the population variance (not sample variance — no Bessel
correction in layer norm).*/
    }
    sdata[threadIdx.x] = var_sum;
    __syncthreads();

    for(int s = blockDim.x / 2; s > 0; s >>= 1){
        if(threadIdx.x < s){
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    float inv_std = rsqrtf(sdata[0] / d_model +eps); // "rsqrtf" is CUDA intrinsic for 1/sqrtf, which is faster than computing sqrtf and then taking reciprocal. inv_std reciprocal std deviation, cuz the normalization step multiplies by it rather than dividing, mult is faster than division on GPU.
    // "sdata[0] / d_model + eps" variance + epsilon, the eps prevents rsqrtf(0).


    for(int i = threadIdx.x; i < d_model; i += blockDim.x) {
    y[i] = gamma[i] * ((x[i] - mean) * inv_std) + beta[i];
    } // (x[i] - mean) * inv_std) normalized value of x[i] -> x_hat , with mean 0 and std 1.
    // gamma[i]*x_hat + beta[i] the affine transform. gamma and beta are indexed by feature dimension i, NOT by token.means all tokens share same learned scale/shift for all the feature.

} 
/*Grid-stride loop: each thread accumulates partial sums of x[i].
Then parallel tree reduction (same as softmax) gives us the total sum
in sdata[0]. Divide by d_model to get the mean.*/

void layernorm_forward(float* output, const float* input, const float* gamma, const float* beta, int N, int d_model, float eps){
     
    int threads = min(256, d_model);// we need at least as many threads as d_model to ensure parallel reduction works, but capped at 256 for occupancy.
    size_t shared = threads * sizeof(float);
    layernorm_fwd_kernel<<<N, threads, shared>>>(output, input, gamma, beta, d_model, eps);
}

__global__ void layernorm_backward_kernel(float* dx, float* dgamma, float* dbeta, const float* dy, const float* x, const float* gamma, int d_model, float eps){
   
    extern __shared__ float sdata[];
    int token = blockIdx.x;
    const float* x_ptr = x + token * d_model;
    const float* dyi = dy + token * d_model;
    float* dxi = dx + token * d_model;

// Recompute mean and inv_std 
    float sum = 0.0f;
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        sum += x_ptr[i];
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1){
        if (threadIdx.x < s){
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    float mean = sdata[0] / d_model;

    float var_sum = 0.0f;
    for(int i = threadIdx.x;i < d_model; i += blockDim.x){
        float diff = x_ptr[i] - mean;
        var_sum += diff * diff;
    }
    sdata[threadIdx.x] = var_sum;
    __syncthreads();

    for(int s = blockDim.x / 2; s > 0; s >>= 1){
        if(threadIdx.x < s){
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    float inv_std = rsqrtf(sdata[0] / d_model + eps);

    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        float x_hat = (x_ptr[i] - mean) * inv_std;
        atomicAdd(&dbeta[i],  dyi[i]);
        atomicAdd(&dgamma[i], dyi[i] * x_hat);
    }

    float sum_dy = 0.0f, sum_dy_xhat = 0.0f;
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        float x_hat = (x_ptr[i] - mean) * inv_std;
        sum_dy      += dyi[i] * gamma[i];
        sum_dy_xhat += dyi[i] * gamma[i] * x_hat;
    }

    // Two separate reductions to compute mean_dy and mean_dy_xhat
    sdata[threadIdx.x] = sum_dy;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    float mean_dy = sdata[0] / d_model;

    sdata[threadIdx.x] = sum_dy_xhat;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    float mean_dy_xhat = sdata[0] / d_model;

    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        float x_hat = (x_ptr[i] - mean) * inv_std;
        dxi[i] = inv_std * (dyi[i] * gamma[i] - mean_dy - x_hat * mean_dy_xhat);
    }


}/*dy is the gradient flowing INTO this layer from
the layer above (in the backward direction). We need to compute dx to
pass the gradient further back.*/

// dgamma and dbeta are NOT per token they're global parameters. thier shape is d_model not Nxd_model.

void layernorm_backward(float* d_input, float* d_gamma, float* d_beta,
                        const float* d_output, const float* input,
                        const float* gamma, int N, int d_model, float eps)
{
    int threads = min(256, d_model);
    size_t shared = threads * sizeof(float);
    layernorm_backward_kernel<<<N, threads, shared>>>(
        d_input, d_gamma, d_beta, d_output, input, gamma, d_model, eps);
}

// read and document you open this next time baby. also verify......
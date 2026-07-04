#include "decoder.cuh"
#include "../kernels/matmul.cuh"
#include <cuda_runtime.h>

// Reassemble patches back into image pixels (exact inverse of extraction)
__global__ void patch_reassembler_kernel(float* image, const float* patches, int batch,
                                          int C, int H, int W, int patch_size, int num_patches_h)
{
    int b = blockIdx.z, p = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int patch_dim = patch_size * patch_size * C;
    if (d >= patch_dim) return;

    int ph = p / num_patches_h, pw = p % num_patches_h;
    int c  = d / (patch_size * patch_size);
    int py = (d % (patch_size * patch_size)) / patch_size;
    int px = d % patch_size;
    int iy = ph * patch_size + py, ix = pw * patch_size + px;

    image[b * C * H * W + c * H * W + iy * W + ix] =
        patches[b * (num_patches_h * num_patches_h) * patch_dim + p * patch_dim + d];
}

__global__ void clamp_kernel(float* x, float lo, float hi, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = fmaxf(lo, fminf(hi, x[i]));
}

// Gradient through clamp: pass through where |x| < bound, zero otherwise
__global__ void clamp_backward_kernel(float* d_in, const float* d_out,
                                       const float* x, float lo, float hi, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d_in[i] = (x[i] > lo && x[i] < hi) ? d_out[i] : 0.0f;
}

// Scatter patches gradient back into (batch*NP × patch_dim) from image gradient
// Exact reverse index mapping of patch_reassembler_kernel
__global__ void patch_scatter_kernel(float* d_patches, const float* d_image, int batch,
                                      int C, int H, int W, int patch_size, int num_patches_h)
{
    int b = blockIdx.z, p = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int patch_dim = patch_size * patch_size * C;
    if (d >= patch_dim) return;

    int ph = p / num_patches_h, pw = p % num_patches_h;
    int c  = d / (patch_size * patch_size);
    int py = (d % (patch_size * patch_size)) / patch_size;
    int px = d % patch_size;
    int iy = ph * patch_size + py, ix = pw * patch_size + px;
    int NP = num_patches_h * num_patches_h;

    d_patches[b * NP * patch_dim + p * patch_dim + d] =
        d_image[b * C * H * W + c * H * W + iy * W + ix];
}

// ── Forward ───────────────────────────────────────────────────────────────────

void decoder_forward(float* output, const float* input, const DecoderParams* params,
                     const EncoderConfig* cfg, int batch)
{
    int NP = cfg->num_patches, PD = cfg->patch_dim;
    int C = cfg->channels, H = cfg->image_size, W = cfg->image_size;
    int P = cfg->patch_size, NPH = H / P;

    float* d_patches;
    cudaMalloc(&d_patches, (size_t)batch * NP * PD * sizeof(float));
    // Project: (batch*NP × d_model) * (d_model × PD) = (batch*NP × PD)
    matrixmult(input, params->proj_W, d_patches, batch * NP, cfg->d_model, PD);

    dim3 grid_re((PD + 255) / 256, NP, batch);
    patch_reassembler_kernel<<<grid_re, 256>>>(output, d_patches, batch, C, H, W, P, NPH);

    cudaFree(d_patches);
}

// ── Backward ──────────────────────────────────────────────────────────────────
// d_output : gradient w.r.t. the reconstructed image (batch × C × H × W)
// d_input  : gradient to pass back to transformer (batch × NP × d_model)
// params->d_proj_W : accumulated weight gradient

void decoder_backward(float* d_input, DecoderParams* params,
                      const float* d_output, const float* output,
                      const float* input, const EncoderConfig* cfg, int batch)
{
    int NP  = cfg->num_patches, PD = cfg->patch_dim;
    int C   = cfg->channels, H = cfg->image_size, W = cfg->image_size;
    int P   = cfg->patch_size, NPH = H / P;

    // ── 2. Scatter image gradient back into patch layout ──────────────────────
    float* d_patches;
    cudaMalloc(&d_patches, (size_t)batch * NP * PD * sizeof(float));
    dim3 grid((PD + 255) / 256, NP, batch);
    patch_scatter_kernel<<<grid, 256>>>(d_patches, d_output, batch, C, H, W, P, NPH);

    // ── 3. proj_W backward ────────────────────────────────────────────────────
    // d_proj_W += input^T * d_patches  (d_model × PD, accumulated)
    matmul_AtB(input, d_patches, params->d_proj_W, batch * NP, cfg->d_model, PD);

    // d_input = d_patches * proj_W^T  (batch*NP × d_model)
    matmul_ABt(d_patches, params->proj_W, d_input, batch * NP, PD, cfg->d_model);
    cudaFree(d_patches);
}
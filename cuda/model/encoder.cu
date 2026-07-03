//image encoder

#include "encoder.cuh"
#include "../kernels/matmul.cuh"
#include <cuda_runtime.h>
#include <math.h>
#include <vector>
#include <iostream>

// Extract image patches into a flat buffer.
// patches[b, p, d] = images[b, c, iy, ix]  where (p, d) → (ph,pw,c,py,px)
__global__ void patch_extractor_kernel(
    float* patches, const float* images, int batch, int C, int H, int W,
    int patch_size, int num_patches_h)
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

    patches[b * (num_patches_h * num_patches_h) * patch_dim + p * patch_dim + d] =
        images[b * C * H * W + c * H * W + iy * W + ix];
}

// Positional encoding add: adds fixed (NP × d_model) embedding to all batch items
__global__ void add_positional_encoding_kernel(float* output, const float* pos_emb,
                                               int batch, int num_patches, int d_model)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * num_patches * d_model) return;
    int p = (idx / d_model) % num_patches;
    output[idx] += pos_emb[p * d_model + (idx % d_model)];
}

// ── Forward ───────────────────────────────────────────────────────────────────

void encoder_forward(float* output, const float* images, const EncoderParams* params,
                     const EncoderConfig* cfg, int batch)
{
    int P = cfg->patch_size, C = cfg->channels;
    int H = cfg->image_size, W = cfg->image_size;
    int NP = cfg->num_patches, PD = cfg->patch_dim;
    int NPH = H / P;

    float* d_patches;
    cudaMalloc(&d_patches, (size_t)batch * NP * PD * sizeof(float));

    dim3 grid_ext((PD + 255) / 256, NP, batch);
    patch_extractor_kernel<<<grid_ext, 256>>>(d_patches, images, batch, C, H, W, P, NPH);

    // Linear projection: (batch*NP × PD) * (PD × d_model) = (batch*NP × d_model)
    matrixmult(d_patches, params->patch_proj_W, output, batch * NP, PD, cfg->d_model);

    int total = batch * NP * cfg->d_model;
    add_positional_encoding_kernel<<<(total+255)/256, 256>>>(
        output, params->pos_embedding, batch, NP, cfg->d_model);

    cudaFree(d_patches);
}

// ── Sinusoidal positional encoding initialization ─────────────────────────────

void encoder_init(EncoderParams* params, const EncoderConfig* cfg) {
    int NP = cfg->num_patches, D = cfg->d_model;
    std::vector<float> pos_enc(NP * D);
    for (int p = 0; p < NP; p++) {
        for (int i = 0; i < D; i += 2) {
            float freq = 1.0f / powf(10000.0f, (float)i / D);
            pos_enc[p * D + i]     = sinf(p * freq);
            if (i + 1 < D)
                pos_enc[p * D + i + 1] = cosf(p * freq);
        }
    }
    cudaMemcpy(params->pos_embedding, pos_enc.data(),
               NP * D * sizeof(float), cudaMemcpyHostToDevice);
}

// ── Backward ──────────────────────────────────────────────────────────────────
// Backpropagates through the patch projection.
// Positional encodings are fixed (sinusoidal) — no gradient needed for them.
// We do NOT backprop into the image pixels (d_images is not used by the decoder).

void encoder_backward(
    float* d_images,              // not used (images are input data, not parameters)
    EncoderParams* params,
    const float* d_output,        // gradient w.r.t. encoder output (batch*NP × d_model)
    const float* images,          // original images (batch × C × H × W), needed to get patches
    const EncoderConfig* cfg,
    int batch_size)
{
    int P   = cfg->patch_size, C = cfg->channels;
    int H   = cfg->image_size, W = cfg->image_size;
    int NP  = cfg->num_patches, PD = cfg->patch_dim;
    int NPH = H / P;

    // Re-extract patches from images to get the forward-pass activations for dW
    float* d_patches;
    cudaMalloc(&d_patches, (size_t)batch_size * NP * PD * sizeof(float));
    dim3 grid_ext((PD + 255) / 256, NP, batch_size);
    patch_extractor_kernel<<<grid_ext, 256>>>(d_patches, images, batch_size, C, H, W, P, NPH);

    // d_patch_proj_W += patches^T * d_output  (PD × d_model, accumulated)
    matmul_AtB(d_patches, d_output, params->d_patch_proj_W,
               batch_size * NP, PD, cfg->d_model);

    cudaFree(d_patches);

    // Zero d_images — we don't backprop into pixel space (images are data, not params)
    if (d_images != nullptr)
        cudaMemset(d_images, 0, (size_t)batch_size * C * H * W * sizeof(float));
}
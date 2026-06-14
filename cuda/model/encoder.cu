//image encoder

#include "encoder.cuh"
#include "../kernels/matmul.cuh"
#include <cuda_runtime.h>
#include <math.h>
#include <vector>
#include <iostream>

__global__ void patch_extractor_kernel(
    float* patches, const float* images, int batch, int C, int H, int W, int patch_size,int num_patches_h)//num_patches_h - num of patches along one dimension(=/patch_size)
    {
        int b = blockIdx.z;// batch item
        int p = blockIdx.y;// patch index
        int d = blockIdx.x * blockDim.x + threadIdx.x;// position in flattened patch.
        // 3D grid with z for batch, y for patch index, x for position within patch.
        //d run from 0 to patch_dim - 1 = p*p*C - 1.

        int patch_dim = patch_size * patch_size * C;
        if (d >= patch_dim) return;// boundary check
        
        int ph = p / num_patches_h;// patch row in the patch grid
        int pw = p % num_patches_h;// patch column in the patch grid

        int c = d / (patch_size * patch_size);// channel index within the patch
        int py = (d % (patch_size * patch_size)) / patch_size;// pixel row y  within the patch
        int px = d % patch_size;// pixel column x within the patch
        //decompose flat index d into channel, within-patch -y, within-patch-x.

        int iy = ph * patch_size + py;// absolute y in the full image
        int ix = pw * patch_size + px;// absolute x in the full image

         patches[b * (num_patches_h * num_patches_h) * patch_dim + p * patch_dim + d] = images[b * C * H * W + c * H * W + iy * W + ix];
         //destination indext, flat row-major for (batch, num_patches, patch_dim), source index : NCHW format- b*C*H*W skips to the right batch item, c*H*W skips to the right channel, iy*w + ix is the 2D pixel position. 

         
    }

__global__  void add_positional_encoding_kernel(float* output, const float* pos_emb, int batch, int num_patches, int d_model){
            
            int idx = blockIdx.x * blockDim.x + threadIdx.x;// idx is a flat index over the entire (batch x num_patches x d_model) output tensor
            if(idx >= batch * num_patches * d_model) return;
            int p = (idx / d_model) % num_patches;// idx/d_model, which token(batch_item*num_patches + patch_index) and then %num_patches strips out the batch dimension to get hte patch position
            output[idx] += pos_emb[p * d_model + (idx % d_model)];// idx % d_model features dimension.
}

void encoder_fwd( float* output, const float* images, const EncoderParams* params, const EncoderConfig* cfg, int batch){
    
   int P = cfg->patch_size, C = cfg->channels;
   int H = cfg->image_size, W = cfg->image_size;
   int NP = cfg->num_patches;
   int PD = cfg->patch_dim;
   // unpack config into local variables for readability.

   float* d_patches;
   cudaMalloc(&d_patches, batch * NP * PD * sizeof(float)); // allocate a temp GPU buffer for extarction patches(batch x num_patch x patch_dim)
   //sizeof(float) = 4bytes so for batch 32, 64 patches, 192 patch_dim: 32x 64x 192x4 = 1.5mb appx.

   int NPH = H / P;
   dim3 grid_ext((PD + 255) / 256, NP, batch);// a 3D grid specfication.
   patch_extractor_kernel<<<grid_ext, 256>>>(d_patches, images, batch, C, H, W, P, NPH);// 256 per block, handling the patch_dim dimention.

   matrixmult(d_patches, params->patch_projection_w, output, batch * NP, PD, cfg->d_model);

    int total = batch * NP * cfg->d_model;// verify this.
    add_positional_encoding_kernel<<<(total+255)/256, 256>>>(
        output, params->pos_embedding, batch, NP, cfg->d_model);

    cudaFree(d_patches);

}

void encoder_init(EncoderParams* params, const EncoderConfig* cfg) {
    int NP = cfg->num_patches;
    int D  = cfg->d_model;
    std::vector<float> pos_enc(NP * D);
    for (int p = 0; p < NP; p++) {
        for (int i = 0; i < D; i += 2) {
            float freq = 1.0f / powf(10000.0f, (float)i / D);
            pos_enc[p * D + i]     = sinf(p * freq);
            pos_enc[p * D + i + 1] = cosf(p * freq);
        }
    }
    cudaMemcpy(params->pos_embedding, pos_enc.data(),
               NP * D * sizeof(float), cudaMemcpyHostToDevice);
}/*Computes sinusoidal positional encodings on the CPU, then copies to GPU.
freq = 1/10000^(i/D) — a different frequency for each pair of dimensions. Low-index dimensions oscillate fast (high frequency), high-index dimensions oscillate slow.
Even dimensions get sin, odd dimensions get cos. This gives every position a unique pattern, and nearby positions have similar encodings.
i += 2 — step by 2 since we fill both i and i+1 each iteration.
cudaMemcpyHostToDevice — copies from the CPU std::vector to the GPU buffer.*/
#include "decoder.cuh"
#include "../kernels/matmul.cuh"
#include <cuda_runtime.h>

__global__ void patch_reassembler_kernel(float* image, const float* patches, int batch,  int C, int H, int W, int patch_size, int num_patches_h){

    int b = blockIdx.z, p = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int patch_dim = patch_size * patch_size * C;

    if (d >= patch_dim) return; //exact grid layout as the extractor kernel 

    int ph = p / num_patches_h, pw = p % num_patches_h;
    int c = d / (patch_size * patch_size);
    int py = (d % (patch_size * patch_size)) / patch_size;
    int px = d % patch_size;
    int iy = ph * patch_size + py, ix = pw * patch_size + px; // same index decomposition as exraction- map patch index and postion back to image coordinatesa.

    image[b * C * H * W + c * H * W + iy * W + ix] = patches[b* (num_patches_h * num_patches_h) * patch_dim + p * patch_dim + d];
//reverse of extraction, wrtite from the patches buffer into the image buffer.


}

__global__ void clamp_kernel(float* x, float lo, float hi, int n){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) x[i] = fmaxf(lo, fminf(hi, x[i]));// fminf - clamp upper bound, fmaxf - clamp lower bound. composed gives clamp(x, lo, hi), callede with lo = -1.0f, hi = 1.0f since images are normalized to [-1,1]


}

void decoder_forward(float* output, const float* input, const DecoderParams* params, const EncoderConfig* cfg, int batch){

    int NP = cfg -> num_patches, PD = cfg -> patch_dim;
    int C = cfg -> channels, H = cfg -> image_size, W = cfg -> image_size;
    int P = cfg -> patch_size, NPH = H / P;
    
    float* d_patches;
    cudaMalloc(&d_patches, batch * NP * PD * sizeof(float));
    matrixmult(input, params -> proj_W, d_patches, batch * NP, cfg -> d_model, PD);// write...

    dim3 grid_re((PD + 255) / 256, NP, batch);
    patch_reassembler_kernel<<<grid_re, 256>>>(output, d_patches, batch, C, H, W, P, NPH);

    int total = batch * C * H * W;
    clamp_kernel<<<(total + 255) / 256, 256>>>(output, -1.0f, 1.0f, total);
    //reasseble and clamp

    cudaFree(d_patches);// free temp patch buffer.
}
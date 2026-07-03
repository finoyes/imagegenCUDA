//tocken sequence -> reconstructed image fro simple linear head
//(batch x num_patches x d_model) -> matrixmult -> (batch x num_patches x patch_dim) -> reassemble -> (batch x C x H x W)


#pragma once
#include <cuda_runtime.h>
#include "encoder.cuh"

struct DecoderParams{
    float* proj_W;// (d_model x patch_dim)
    float* proj_b;// (patch_dim,)
    float* d_proj_W;
    float* d_proj_b;

};// mirror of the encoder's patch projection, but in reverse: d_model -> patch_dim.
//proj_W is d_model x patch_dim, while encoder's patch_proj_w is (patch_dim x d_model) - transposed.

// Forward: token sequence -> reconstructed image
// input:  (batch x num_patches x d_model)
// output: (batch x C x H x W)
void decoder_forward(
    float* output,
    const float* input,
    const DecoderParams* params,
    const EncoderConfig* cfg,
    int batch
);

void decoder_backward(
    float* d_input,
    DecoderParams* params,
    const float* d_output,   // gradient w.r.t. reconstructed image
    const float* output,     // forward output (clamped image) — needed for clamp backward
    const float* input,      // forward input (token sequence) — needed for dW
    const EncoderConfig* cfg,
    int batch
);
// the encoder converts a (channel x height x widht) image into a (num_patched x d_model) token:
//split image into PxP patches -> num_patches = (H/P)^2 patches, each of size PxPxC = patch_dim
//linear projection, each patch patch_dim -> d_model. add positional encoding.

#pragma once
#include <cuda_runtime.h>

struct EncoderParams{

float* patch_proj_W; // maps patch_dim * d_model projection. linear projection layer
float* patch_proj_b; //(d_model) bias
float* pos_embedding;// (num_patched x d_model). one d_model Dim vec per patch position, learned; 
float* cls_token;// ( d_model,)
float* d_patch_proj_W;
float* d_patch_proj_b;
float* d_pos_embedding;
};

struct EncoderConfig{

int image_size;// 
int patch_size;
int channels;
int d_model;
int num_patches;// (image_size/patch_size)^2
int patch_dim;// patch_size * patch_size * channels

};// pure config as no GPU pointers

// Forward: images -> token sequence
// images: (batch x C x H x W) device pointer
// output: (batch x num_patches x d_model)
void encoder_forward(
    float* output,
    const float* images,
    const EncoderParams* params,
    const EncoderConfig* cfg,
    int batch_size
);

void encoder_backward(
    float* d_images,
    EncoderParams* params,
    const float* d_output,
    const float* images,
    const EncoderConfig* cfg,
    int batch_size
);

// Initialize with Kaiming uniform (patches) and sinusoidal (positions)
void encoder_init(EncoderParams* params, const EncoderConfig* cfg);

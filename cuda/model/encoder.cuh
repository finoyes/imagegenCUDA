// the encoder converts a (channel x height x widht) image into a (num_patched x d_model) token:
//split image into PxP patches -> num_patches = (H/P)^2 patches, each of size PxPxC = patch_dim
//linear projection, each patch patch_dim -> d_model. add positional encoding.

#pragma once
#include <cuda_runtime.h>

struct EncoderParams{

float* patch_projection_w; // maps patch_dim * d_model projection. linear projection layer
float* patch_projection_b; //(d_model) bias
float* pos_embedding;// (num_patched x d_model). one d_model Dim vec per patch position, learned; 
float* cls_token;// ( d_model,)
float* d_patch_projection_W;
float* d_patch_projection_b;
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


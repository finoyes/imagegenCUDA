/*
   x = x + attention(layernorm(x)) - sublayer1
   x = x + FFN(layernorm(x)) - sublayer2
*/

#pragma once
#include "decoder.cuh"
#include <cuda_runtime.h>
#include "../kernels/attention.cuh"
#include "../kernels/layernorm.cuh"

struct FFNParams {
    float* W1; float* b1;   // (d_model x d_ff), (d_ff,)
    float* W2; float* b2;   // (d_ff x d_model), (d_model,)
    float* dW1; float* db1;
    float* dW2; float* db2;
};/*The FFN expands then contracts: d_model → d_ff → d_model.
d_ff = 4 * d_model typically (so 1024 for d_model=256).
The expansion creates a "bottleneck" where the model can mix features nonlinearly.
*/

struct TransformerBlockParams {
    AttentionParams attn;
    FFNParams ffn;
    float* ln1_gamma; float* ln1_beta;
    float* ln2_gamma; float* ln2_beta;
    float* d_ln1_gamma; float* d_ln1_beta;
    float* d_ln2_gamma; float* d_ln2_beta;
};// two layernorm per block one before attention and one before FFN each norm has it's own gamma and beta.

struct TransformerParams {
    TransformerBlockParams* blocks;   // array of num_layers blocks
    int num_layers;
    int d_model;
    int num_heads;
    int d_ff;
};
struct TransformerCache {
    float** ln1_out;        // (num_layers) layer-norm 1 outputs
    float** attn_out;       // (num_layers) attention sublayer outputs
    float** ln2_out;        // (num_layers) layer-norm 2 outputs
    float** ffn_mid;        // (num_layers) post-GeLU FFN intermediate (d_ff)
    float** ffn_pre_gelu;   // (num_layers) pre-GeLU FFN intermediate — needed for GeLU backward
    float** x_after_attn;   // (num_layers) residual after attention (= input to LN2) — needed for LN2 backward
    AttentionCache* attn_caches;
};

void transformer_forward(
    float* output,
    TransformerCache* cache,
    const float* input,
    const TransformerParams* params,
    int batch, int seq_len
);

void transformer_backward(
    float* d_input,
    TransformerParams* params,
    const float* d_output,
    const TransformerCache* cache,
    int batch, int seq_len
);
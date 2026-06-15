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

struct TransformerCache {
    float** ln1_out;
    float** attn_out;
    float** ln2_out;
    float** ffn_mid;
    AttentionCache* attn_caches;
};/*float** — array of pointers. ln1_out[l] gives the layer-norm output for layer l.
ffn_mid — the intermediate activation between the two FFN linear layers (after GELU). Needed for backward through GELU.
One AttentionCache per layer.*/
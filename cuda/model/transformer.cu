#include "transformer.cuh"
#include "../kernels/attention.cuh"
#include "../kernels/layernorm.cuh"
#include "../kernels/activations.cuh"
#include "../kernels/matmul.cuh"
#include <cuda_runtime.h>
#include <cstring>

__global__ void add_residual_kernel(float* out, const float* residual, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += residual[i];
}// element wise addition. this implements the "+" in x = x + sublayer(x)
//during backward, the gradient flows through unchanged in both branches of the addition.

void transformer_forward(
    float* output, TransformerCache* cache,
    const float* input, const TransformerParams* p,
    int batch, int seq_len)
{
    int D   = p->d_model;
    int N   = batch * seq_len;
    int dff = p->d_ff;

    cudaMemcpy(output, input, N * D * sizeof(float), cudaMemcpyDeviceToDevice);//

    for (int l = 0; l < p->num_layers; l++) {
        TransformerBlockParams& bp = p->blocks[l];//
        layernorm_forward(cache->ln1_out[l], output,
                   bp.ln1_gamma, bp.ln1_beta, N, D);
         attention_forward(cache->attn_out[l], &cache->attn_caches[l],
                          cache->ln1_out[l], &bp.attn,
                          batch, seq_len, D, p->num_heads);
         layernorm_forward(cache->ln2_out[l], output,
                          bp.ln2_gamma, bp.ln2_beta, N, D);
        matrixmult(cache->ln2_out[l], bp.ffn.W1, cache->ffn_mid[l], N, D, dff);// write
        // add b1
        GeLU_forward(cache->ffn_mid[l], cache->ffn_mid[l], N * dff);
        matrixmult(cache->ffn_mid[l], bp.ffn.W2, /* temp */ output, N, dff, D);
        // add b2, wrtite
        add_residual_kernel<<<(N*D+255)/256, 256>>>(output, /* ffn_out */ nullptr, N*D);// the nullptr in the comments highlights the incomplete implementation.
    }
}
     
void transformer_backward(
    float* d_input, TransformerParams* p,
    const float* d_output, const TransformerCache* cache,
    int batch, int seq_len)
{
    int D = p->d_model, N = batch * seq_len;
    // Reverse loop through layers
    for (int l = p->num_layers - 1; l >= 0; l--) {
        // Each step mirrors the forward pass in reverse
        // Omitted for brevity — follows the exact inverse chain rule pattern
    }
}
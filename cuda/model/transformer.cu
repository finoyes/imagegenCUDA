#include "transformer.cuh"
#include "../kernels/attention.cuh"
#include "../kernels/layernorm.cuh"
#include "../kernels/activations.cuh"
#include "../kernels/matmul.cuh"
#include <cuda_runtime.h>
#include <cstring>

// Element-wise residual add: out += residual
// During backward, the gradient flows through unchanged in both branches.
__global__ void add_residual_kernel(float* out, const float* residual, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += residual[i];
}

// Broadcast bias (cols,) across all (rows × cols)
__global__ void add_bias_kernel(float* out, const float* bias, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * cols) out[i] += bias[i % cols];
}

// Accumulate bias gradient: db[j] += sum_rows d[*, j]
__global__ void bias_grad_tr_kernel(float* db, const float* d, int rows, int cols) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cols) return;
    float s = 0.0f;
    for (int r = 0; r < rows; r++) s += d[r * cols + j];
    atomicAdd(&db[j], s);
}

// ── Forward pass ──────────────────────────────────────────────────────────────

void transformer_forward(
    float* output, TransformerCache* cache,
    const float* input, const TransformerParams* p,
    int batch, int seq_len)
{
    int D   = p->d_model;
    int N   = batch * seq_len;
    int dff = p->d_ff;

    // output starts as a copy of input; each layer modifies it in-place
    cudaMemcpy(output, input, (size_t)N * D * sizeof(float), cudaMemcpyDeviceToDevice);

    for (int l = 0; l < p->num_layers; l++) {
        TransformerBlockParams& bp = p->blocks[l];

        // ── Save layer input (needed for LN1 backward) ────────────────────────
        cudaMemcpy(cache->attn_caches[l].input, output,
                   (size_t)N * D * sizeof(float), cudaMemcpyDeviceToDevice);

        // ── Sublayer 1: Attention ─────────────────────────────────────────────
        layernorm_forward(cache->ln1_out[l], output,
                          bp.ln1_gamma, bp.ln1_beta, N, D);
        attention_forward(cache->attn_out[l], &cache->attn_caches[l],
                          cache->ln1_out[l], &bp.attn,
                          batch, seq_len, D, p->num_heads);
        add_residual_kernel<<<(N*D+255)/256, 256>>>(output, cache->attn_out[l], N*D);

        // ── Save post-attention residual (input to LN2) ───────────────────────
        cudaMemcpy(cache->x_after_attn[l], output,
                   (size_t)N * D * sizeof(float), cudaMemcpyDeviceToDevice);

        // ── Sublayer 2: FFN ───────────────────────────────────────────────────
        layernorm_forward(cache->ln2_out[l], output,
                          bp.ln2_gamma, bp.ln2_beta, N, D);

        // W1: (D→dff), save pre-GeLU for backward
        matrixmult(cache->ln2_out[l], bp.ffn.W1, cache->ffn_pre_gelu[l], N, D, dff);
        add_bias_kernel<<<(N*dff+255)/256, 256>>>(cache->ffn_pre_gelu[l], bp.ffn.b1, N, dff);

        // GeLU in-place → ffn_mid holds post-GeLU
        cudaMemcpy(cache->ffn_mid[l], cache->ffn_pre_gelu[l],
                   (size_t)N * dff * sizeof(float), cudaMemcpyDeviceToDevice);
        GeLU_forward(cache->ffn_mid[l], cache->ffn_mid[l], N * dff);

        // W2: (dff→D)
        float* ffn_out;
        cudaMalloc(&ffn_out, (size_t)N * D * sizeof(float));
        matrixmult(cache->ffn_mid[l], bp.ffn.W2, ffn_out, N, dff, D);
        add_bias_kernel<<<(N*D+255)/256, 256>>>(ffn_out, bp.ffn.b2, N, D);
        add_residual_kernel<<<(N*D+255)/256, 256>>>(output, ffn_out, N*D);
        cudaFree(ffn_out);
    }
}

// ── Backward pass ─────────────────────────────────────────────────────────────
// All weight gradients are ACCUMULATED (+=). Zero them before the step.
// d_output is the upstream gradient entering from the decoder side.
// d_input receives the gradient to pass back to the encoder.

void transformer_backward(
    float* d_input, TransformerParams* p,
    const float* d_output, const TransformerCache* cache,
    int batch, int seq_len)
{
    int D   = p->d_model;
    int N   = batch * seq_len;
    int dff = p->d_ff;

    // Working gradient buffer — starts as a copy of the upstream gradient,
    // gets propagated backwards through layers.
    float* d_cur;
    cudaMalloc(&d_cur, (size_t)N * D * sizeof(float));
    cudaMemcpy(d_cur, d_output, (size_t)N * D * sizeof(float), cudaMemcpyDeviceToDevice);

    for (int l = p->num_layers - 1; l >= 0; l--) {
        TransformerBlockParams& bp = p->blocks[l];

        // ────────────────────────────────────────────────────────────────────
        // FFN sublayer backward
        // Forward: x_out = x_after_attn + FFN(LN2(x_after_attn))
        // ────────────────────────────────────────────────────────────────────

        // Residual split: d_cur flows back through both branches
        // Branch 2 (FFN): d_ffn_out = d_cur
        // Branch 1 (identity): gradient passes through unchanged → handled below

        // W2 backward
        // d_ffn_mid = d_cur * W2^T  (dff-dimensional)
        float* d_ffn_mid;
        cudaMalloc(&d_ffn_mid, (size_t)N * dff * sizeof(float));
        matmul_ABt(d_cur, bp.ffn.W2, d_ffn_mid, N, D, dff);
        // dW2 += ffn_mid^T * d_cur
        matmul_AtB(cache->ffn_mid[l], d_cur, bp.ffn.dW2, N, dff, D);
        // db2 += rowsum(d_cur)
        bias_grad_tr_kernel<<<(D+255)/256, 256>>>(bp.ffn.db2, d_cur, N, D);

        // GeLU backward (needs pre-GeLU values saved in ffn_pre_gelu)
        float* d_pre_gelu;
        cudaMalloc(&d_pre_gelu, (size_t)N * dff * sizeof(float));
        GeLU_backward(d_pre_gelu, d_ffn_mid, cache->ffn_pre_gelu[l], N * dff);
        cudaFree(d_ffn_mid);

        // W1 backward
        // dW1 += ln2_out^T * d_pre_gelu
        matmul_AtB(cache->ln2_out[l], d_pre_gelu, bp.ffn.dW1, N, D, dff);
        // db1 += rowsum(d_pre_gelu)
        bias_grad_tr_kernel<<<(dff+255)/256, 256>>>(bp.ffn.db1, d_pre_gelu, N, dff);
        // d_ln2_out = d_pre_gelu * W1^T
        float* d_ln2_out;
        cudaMalloc(&d_ln2_out, (size_t)N * D * sizeof(float));
        matmul_ABt(d_pre_gelu, bp.ffn.W1, d_ln2_out, N, dff, D);
        cudaFree(d_pre_gelu);

        // LN2 backward — input to LN2 was x_after_attn[l]
        float* d_x_after_attn;
        cudaMalloc(&d_x_after_attn, (size_t)N * D * sizeof(float));
        layernorm_backward(d_x_after_attn, bp.d_ln2_gamma, bp.d_ln2_beta,
                           d_ln2_out, cache->x_after_attn[l], bp.ln2_gamma, N, D);
        cudaFree(d_ln2_out);

        // Residual add backward: d_x_after_attn += d_cur (identity branch)
        vec_add_inplace(d_x_after_attn, d_cur, N * D);

        // d_cur now propagates back from x_after_attn
        cudaMemcpy(d_cur, d_x_after_attn, (size_t)N * D * sizeof(float),
                   cudaMemcpyDeviceToDevice);
        cudaFree(d_x_after_attn);

        // ────────────────────────────────────────────────────────────────────
        // Attention sublayer backward
        // Forward: x_after_attn = x_in + Attn(LN1(x_in))
        // ────────────────────────────────────────────────────────────────────

        // Attention backward — input to attention was ln1_out[l]
        float* d_ln1_out;
        cudaMalloc(&d_ln1_out, (size_t)N * D * sizeof(float));
        attention_backward(d_ln1_out,
                           &bp.attn, d_cur,
                           &cache->attn_caches[l],
                           batch, seq_len, D, p->num_heads);

        // LN1 backward — input to LN1 was attn_caches[l].input (= layer x_in)
        float* d_x_in;
        cudaMalloc(&d_x_in, (size_t)N * D * sizeof(float));
        layernorm_backward(d_x_in, bp.d_ln1_gamma, bp.d_ln1_beta,
                           d_ln1_out, cache->attn_caches[l].input, bp.ln1_gamma, N, D);
        cudaFree(d_ln1_out);

        // Residual add backward: d_x_in += d_cur (identity branch)
        vec_add_inplace(d_x_in, d_cur, N * D);

        // d_cur = gradient wrt layer input, becomes upstream for next lower layer
        cudaMemcpy(d_cur, d_x_in, (size_t)N * D * sizeof(float),
                   cudaMemcpyDeviceToDevice);
        cudaFree(d_x_in);
    }

    // Copy final gradient to d_input (to pass back to encoder)
    cudaMemcpy(d_input, d_cur, (size_t)N * D * sizeof(float),
               cudaMemcpyDeviceToDevice);
    cudaFree(d_cur);
}
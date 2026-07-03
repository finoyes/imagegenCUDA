#include "attention.cuh"
#include "matmul.cuh"
#include "softmax.cuh"
#include <cuda_runtime.h>
#include <math.h>

// ── Helper kernels ────────────────────────────────────────────────────────────

// Broadcast bias (d_model,) across (rows × d_model)
__global__ void bias_summ_krln(float* out, const float* bias, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * cols) out[i] += bias[i % cols];
}

// Accumulate bias gradient: db[j] += sum_rows d[*, j]
__global__ void bias_grad_kernel(float* db, const float* d, int rows, int cols) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cols) return;
    float s = 0.0f;
    for (int r = 0; r < rows; r++) s += d[r * cols + j];
    atomicAdd(&db[j], s);
}

// Scale all elements by scalar
__global__ void scale_kernel(float* x, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= scale;
}

// Permute (batch, seq, heads, d_k) → (batch*heads, seq, d_k)
// in[b, s, h, k]  →  out[(b*heads+h), s, k]
__global__ void bshd_to_bhsd_kernel(float* out, const float* in,
                                     int batch, int seq, int heads, int d_k)
{
    int b  = blockIdx.z;
    int h  = blockIdx.y;
    int sd = blockIdx.x * blockDim.x + threadIdx.x;
    if (sd >= seq * d_k) return;
    int s = sd / d_k, d = sd % d_k;
    out[(b * heads + h) * seq * d_k + s * d_k + d] =
        in[b * seq * heads * d_k + s * heads * d_k + h * d_k + d];
}

// Permute (batch*heads, seq, d_k) → (batch, seq, heads, d_k)  [inverse]
// in[(b*heads+h), s, k]  →  out[b, s, h, k]
__global__ void bhsd_to_bshd_kernel(float* out, const float* in,
                                     int batch, int seq, int heads, int d_k)
{
    int b  = blockIdx.z;
    int h  = blockIdx.y;
    int sd = blockIdx.x * blockDim.x + threadIdx.x;
    if (sd >= seq * d_k) return;
    int s = sd / d_k, d = sd % d_k;
    out[b * seq * heads * d_k + s * heads * d_k + h * d_k + d] =
        in[(b * heads + h) * seq * d_k + s * d_k + d];
}

// Transpose last two dims: (BH, A, B) → (BH, B, A)
__global__ void transpose_last2_kernel(float* out, const float* in,
                                        int BH, int A, int B)
{
    int bh = blockIdx.z;
    int i  = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A * B) return;
    int a = i / B, b = i % B;
    out[bh * B * A + b * A + a] = in[bh * A * B + a * B + b];
}

// ── Forward pass ──────────────────────────────────────────────────────────────

void attention_forward(float* output, AttentionCache* cache, const float* input,
                       const AttentionParams* p,
                       int batch, int seq_len, int d_model, int num_heads)
{
    int d_k    = d_model / num_heads;
    int BH     = batch * num_heads;
    int total  = batch * seq_len;
    int threads = 256;
    int n_qkv  = total * d_model;

    // Save input for backward
    cudaMemcpy(cache->input, input, (size_t)n_qkv * sizeof(float),
               cudaMemcpyDeviceToDevice);

    // ── Project Q, K, V ──────────────────────────────────────────────────────
    matrixmult(input, p->W_Q, cache->Q, total, d_model, d_model);
    matrixmult(input, p->W_K, cache->K, total, d_model, d_model);
    matrixmult(input, p->W_V, cache->V, total, d_model, d_model);
    bias_summ_krln<<<(n_qkv+threads-1)/threads, threads>>>(cache->Q, p->b_Q, total, d_model);
    bias_summ_krln<<<(n_qkv+threads-1)/threads, threads>>>(cache->K, p->b_K, total, d_model);
    bias_summ_krln<<<(n_qkv+threads-1)/threads, threads>>>(cache->V, p->b_V, total, d_model);

    // ── Head permute: (batch, seq, heads, d_k) → (BH, seq, d_k) ─────────────
    dim3 perm_grid((seq_len * d_k + threads - 1) / threads, num_heads, batch);
    bshd_to_bhsd_kernel<<<perm_grid, threads>>>(cache->Q_h, cache->Q, batch, seq_len, num_heads, d_k);
    bshd_to_bhsd_kernel<<<perm_grid, threads>>>(cache->K_h, cache->K, batch, seq_len, num_heads, d_k);
    bshd_to_bhsd_kernel<<<perm_grid, threads>>>(cache->V_h, cache->V, batch, seq_len, num_heads, d_k);

    // ── Transpose K_h: (BH, seq, d_k) → (BH, d_k, seq) ─────────────────────
    dim3 trans_grid((seq_len * d_k + threads - 1) / threads, 1, BH);
    transpose_last2_kernel<<<trans_grid, threads>>>(cache->Kt_h, cache->K_h, BH, seq_len, d_k);

    // ── Scores = Q_h * Kt_h / sqrt(d_k): (BH, seq, seq) ─────────────────────
    batched_matrixmult(cache->Q_h, cache->Kt_h, cache->scores, BH, seq_len, d_k, seq_len);
    float scale = 1.0f / sqrtf((float)d_k);
    int n_scores = BH * seq_len * seq_len;
    scale_kernel<<<(n_scores+threads-1)/threads, threads>>>(cache->scores, scale, n_scores);

    // ── Softmax over key dimension (in-place) ─────────────────────────────────
    attention_softmax(cache->scores, batch, num_heads, seq_len);

    // ── Weighted values: (BH, seq, seq) × (BH, seq, d_k) = (BH, seq, d_k) ───
    float* attn_head;
    cudaMalloc(&attn_head, (size_t)BH * seq_len * d_k * sizeof(float));
    batched_matrixmult(cache->scores, cache->V_h, attn_head, BH, seq_len, seq_len, d_k);

    // ── Un-permute: (BH, seq, d_k) → (total, d_model) ───────────────────────
    bhsd_to_bshd_kernel<<<perm_grid, threads>>>(cache->proj_in, attn_head, batch, seq_len, num_heads, d_k);
    cudaFree(attn_head);

    // ── Output projection ─────────────────────────────────────────────────────
    matrixmult(cache->proj_in, p->W_O, output, total, d_model, d_model);
    bias_summ_krln<<<(n_qkv+threads-1)/threads, threads>>>(output, p->b_O, total, d_model);
}

// ── Backward pass ─────────────────────────────────────────────────────────────
// Weight gradients are ACCUMULATED (+= into p->dW_*, p->db_*).
// d_input receives the gradient passed back to the previous layer (overwritten).

void attention_backward(float* d_input, AttentionParams* p,
                        const float* d_output, const AttentionCache* cache,
                        int batch, int seq_len, int d_model, int num_heads)
{
    int d_k    = d_model / num_heads;
    int BH     = batch * num_heads;
    int total  = batch * seq_len;
    int threads = 256;
    int n_qkv  = total * d_model;

    // ── 1. W_O backward ───────────────────────────────────────────────────────
    // dW_O += proj_in^T * d_output  (d_model×d_model, accumulated)
    matmul_AtB(cache->proj_in, d_output, p->dW_O, total, d_model, d_model);
    // db_O += rowsum(d_output)
    bias_grad_kernel<<<(d_model+threads-1)/threads, threads>>>(p->db_O, d_output, total, d_model);
    // d_proj_in = d_output * W_O^T
    float* d_proj_in;
    cudaMalloc(&d_proj_in, (size_t)n_qkv * sizeof(float));
    matmul_ABt(d_output, p->W_O, d_proj_in, total, d_model, d_model);

    // ── 2. Permute d_proj_in → (BH, seq, d_k) ────────────────────────────────
    float* d_attn_head;
    cudaMalloc(&d_attn_head, (size_t)BH * seq_len * d_k * sizeof(float));
    dim3 perm_grid((seq_len * d_k + threads - 1) / threads, num_heads, batch);
    bshd_to_bhsd_kernel<<<perm_grid, threads>>>(d_attn_head, d_proj_in, batch, seq_len, num_heads, d_k);
    cudaFree(d_proj_in);

    // ── 3. scores * V_h backward ──────────────────────────────────────────────
    // d_scores_w = d_attn_head * V_h^T: (BH, seq, d_k) × (BH, d_k, seq) = (BH, seq, seq)
    float* d_scores_w;
    cudaMalloc(&d_scores_w, (size_t)BH * seq_len * seq_len * sizeof(float));
    batched_matmul_ABt(d_attn_head, cache->V_h, d_scores_w, BH, seq_len, d_k, seq_len);

    // d_V_h = scores^T * d_attn_head: (BH, seq, seq)^T × (BH, seq, d_k) = (BH, seq, d_k)
    float* d_V_h;
    cudaMalloc(&d_V_h, (size_t)BH * seq_len * d_k * sizeof(float));
    cudaMemset(d_V_h, 0, (size_t)BH * seq_len * d_k * sizeof(float));
    batched_matmul_AtB(cache->scores, d_attn_head, d_V_h, BH, seq_len, seq_len, d_k);
    cudaFree(d_attn_head);

    // ── 4. Softmax backward ───────────────────────────────────────────────────
    float* d_scores_pre;
    cudaMalloc(&d_scores_pre, (size_t)BH * seq_len * seq_len * sizeof(float));
    attention_softmax_backward(d_scores_pre, cache->scores, d_scores_w,
                               batch, num_heads, seq_len);
    cudaFree(d_scores_w);

    // Scale by 1/sqrt(d_k)
    int n_scores = BH * seq_len * seq_len;
    scale_kernel<<<(n_scores+threads-1)/threads, threads>>>(
        d_scores_pre, 1.0f / sqrtf((float)d_k), n_scores);

    // ── 5. Q * K^T backward ───────────────────────────────────────────────────
    // d_Q_h = d_scores_pre * K_h: (BH, seq, seq) × (BH, seq, d_k) = (BH, seq, d_k)
    float* d_Q_h;
    cudaMalloc(&d_Q_h, (size_t)BH * seq_len * d_k * sizeof(float));
    batched_matrixmult(d_scores_pre, cache->K_h, d_Q_h, BH, seq_len, seq_len, d_k);

    // d_K_h = d_scores_pre^T * Q_h: (BH, seq, seq)^T × (BH, seq, d_k) = (BH, seq, d_k)
    float* d_K_h;
    cudaMalloc(&d_K_h, (size_t)BH * seq_len * d_k * sizeof(float));
    cudaMemset(d_K_h, 0, (size_t)BH * seq_len * d_k * sizeof(float));
    batched_matmul_AtB(d_scores_pre, cache->Q_h, d_K_h, BH, seq_len, seq_len, d_k);
    cudaFree(d_scores_pre);

    // ── 6. Un-permute head-split gradients → (total, d_model) ────────────────
    float* d_Q, *d_K, *d_V;
    cudaMalloc(&d_Q, (size_t)n_qkv * sizeof(float));
    cudaMalloc(&d_K, (size_t)n_qkv * sizeof(float));
    cudaMalloc(&d_V, (size_t)n_qkv * sizeof(float));
    bhsd_to_bshd_kernel<<<perm_grid, threads>>>(d_Q, d_Q_h, batch, seq_len, num_heads, d_k);
    bhsd_to_bshd_kernel<<<perm_grid, threads>>>(d_K, d_K_h, batch, seq_len, num_heads, d_k);
    bhsd_to_bshd_kernel<<<perm_grid, threads>>>(d_V, d_V_h, batch, seq_len, num_heads, d_k);
    cudaFree(d_Q_h); cudaFree(d_K_h); cudaFree(d_V_h);

    // ── 7. Q/K/V projection backward ─────────────────────────────────────────
    // dW_{Q,K,V} += input^T * d_{Q,K,V}
    matmul_AtB(cache->input, d_Q, p->dW_Q, total, d_model, d_model);
    matmul_AtB(cache->input, d_K, p->dW_K, total, d_model, d_model);
    matmul_AtB(cache->input, d_V, p->dW_V, total, d_model, d_model);
    // db_{Q,K,V} += rowsum(d_{Q,K,V})
    bias_grad_kernel<<<(d_model+threads-1)/threads, threads>>>(p->db_Q, d_Q, total, d_model);
    bias_grad_kernel<<<(d_model+threads-1)/threads, threads>>>(p->db_K, d_K, total, d_model);
    bias_grad_kernel<<<(d_model+threads-1)/threads, threads>>>(p->db_V, d_V, total, d_model);

    // d_input = d_Q * W_Q^T + d_K * W_K^T + d_V * W_V^T
    matmul_ABt(d_Q, p->W_Q, d_input, total, d_model, d_model);  // overwrite

    float* tmp;
    cudaMalloc(&tmp, (size_t)n_qkv * sizeof(float));

    matmul_ABt(d_K, p->W_K, tmp, total, d_model, d_model);
    vec_add_inplace(d_input, tmp, n_qkv);

    matmul_ABt(d_V, p->W_V, tmp, total, d_model, d_model);
    vec_add_inplace(d_input, tmp, n_qkv);

    cudaFree(tmp);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
}

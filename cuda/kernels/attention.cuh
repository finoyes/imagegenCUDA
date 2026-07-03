#pragma once

/*
 * Multi-head self-attention:
 *
 *   Q = X*W_Q + b_Q,  K = X*W_K + b_K,  V = X*W_V + b_V
 *   scores = softmax( Q_h * K_h^T / sqrt(d_k) )      per head
 *   out    = concat(scores * V_h) * W_O + b_O
 *
 * Q, K, V are first projected to (total × d_model), then split into
 * num_heads heads of size d_k = d_model/num_heads.
 */

struct AttentionParams {
    float* W_Q; float* W_K; float* W_V; float* W_O;  // (d_model × d_model)
    float* b_Q; float* b_K; float* b_V; float* b_O;  // (d_model,)
    float* dW_Q; float* dW_K; float* dW_V; float* dW_O; // gradient counterparts
    float* db_Q; float* db_K; float* db_V; float* db_O;
};

struct AttentionCache {
    // ── post-projection activations (total × d_model) ──────────────────────
    float* Q;
    float* K;
    float* V;
    // ── head-permuted: (batch × heads × seq × d_k) ──────────────────────────
    // Stored as flat (batch*heads × seq × d_k), head-major layout.
    float* Q_h;
    float* K_h;
    float* V_h;
    float* Kt_h;  // K transposed to (batch*heads × d_k × seq) for Q*K^T
    // ── attention weights after softmax (batch*heads × seq × seq) ───────────
    float* scores;
    // ── input to this attention layer (total × d_model) ─────────────────────
    float* input;
    // ── post-concat, pre-W_O output (total × d_model) ───────────────────────
    float* proj_in;
};

void attention_forward(
    float* output,            // (batch × seq × d_model)
    AttentionCache* cache,    // filled in for backward use
    const float* input,       // (batch × seq × d_model)
    const AttentionParams* p,
    int batch, int seq_len, int d_model, int num_heads
);

void attention_backward(
    float* d_input,           // gradient passed back to previous layer
    AttentionParams* p,
    const float* d_output,    // upstream gradient (batch × seq × d_model)
    const AttentionCache* cache,
    int batch, int seq_len, int d_model, int num_heads
);
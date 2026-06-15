//kernel implementation for the attention mechanism of the transformer.

/* self attention for one head -

Q = X*W_Q, K = X*W_K, V = X*W_V, Q = query, K = key, V = value, where W_Q, W_K, W_V are learnable weight matrices. X is the input sequence (batch x seq_len x d_model).
scores = QxK^T / sqrt(d_k)  (d_k = dimension of K, scaling factor to prevent large dot product values)
weights = softmax(scores) (haha full circle -> softmax converts attention scores to probs/weights)
output = weights x V (weighted sum of values based on attention weights)



Multi-head self attentions => do this H(head.. hehe) times in parallel with different W_Q, W_K, W_V for each head, then concat outputs and project with W_O to get final output of the multi-head attention layer.
*/

struct AttentionParams{
    float* W_Q;// (d_model x d_model)
    float* W_K;
    float* W_V;
    float* W_O;
    float* b_Q;// (d_model,) one dimensional bias.
    float* b_K;
    float* b_V;
    float* b_O;
    float* dW_Q; float* dW_K; float* dW_V; float* dW_O;
    float* db_Q; float* db_K; float* db_V; float* db_O;
}; //dW_x derivative counteparts, for backward pass.

struct AttentionCache{
    float* Q;// (batch x seq_len x d_model)
    float* K;
    float* V;
    float* scores;// (batch x head x seq_len x seq_len)
    float* input;// (batch x seq_len x seq_len)
    
}; // computed during fwd pass and must be kept alive till backward.
// "scores" is the post-attention weights

void attention_forward(
    float* output,              // (batch x seq x d_model)
    AttentionCache* cache,      // filled in for backward use
    const float* input,         // (batch x seq x d_model)
    const AttentionParams* p,
    int batch, int seq_len, int d_model, int num_heads
);

void attention_backward(
    float* d_input,
    AttentionParams* p,
    const float* d_output,
    const AttentionCache* cache,
    int batch, int seq_len, int d_model, int num_heads
);
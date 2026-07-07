---
license: mit
---

# Implementing Transformer from Scratch: A Step-by-Step Guide

This repository provides a detailed guide and implementation of the Transformer architecture from the ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) paper. The implementation focuses on understanding each component through clear code, comprehensive testing, and visual aids.

For implementions of more recent architectural innovations from DeepSeek, see the **Related Implementations** section.

## Table of Contents
1. [Summary and Key Insights](## summary-and-key-insights)
2. [Implementation Details](# implementation-details)
   - [Embedding and Positional Encoding](### embedding-and-positional-encoding)
   - [Transformer Attention](### transformer-attention)
   - [Feed-Forward Network](### feed-forward-network)
   - [Transformer Decoder](### transformer-decoder)
   - [Encoder-Decoder Stack](### encoder-decoder-stack)
   - [Full Transformer](### full-transformer)
   - [Testing](### testing)
   - [Visualizations](### visualizations)
3. [PyData Global 2025 Presentation](## PyData Global 2025 Presentation)
4. [Related Implementations](##Related Implementations)


## Quick Start
View the complete implementation and tutorial in the [Jupyter notebook](Transformer_Implementation_Tutorial.ipynb).

## Summary and Key Insights

### Paper Reference
- ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- Key sections: 
  - 3.1: Encoder and Decoder Stacks
  - 3.2: Attention Mechanism
  - 3.3: Position-wise Feed-Forward Networks
  - 3.4: Embeddings and Softmax
  - 3.5: Positional Encoding
  - 5.4: Regularization (dropout strategy)

### Implementation Strategy
Breaking down the architecture into manageable pieces and gradually adding complexity:

1. Start with foundational components:
   - Embedding + Positional Encoding
   - Single-head self-attention
   
2. Build up attention mechanism:
   - Extend to multi-head attention
   - Add cross-attention capability
   - Implement attention masking
   
3. Construct larger components:
   - Encoder (self-attention + FFN)
   - Decoder (masked self-attention + cross-attention + FFN)
   
4. Combine into final architecture:
   - Encoder-Decoder stack
   - Full Transformer with input/output layers

### Development Tips
1. Visualization and Planning:
   - Draw out tensor dimensions on paper
   - Sketch attention patterns and masks
   - Map each component back to paper equations
   - This helps catch dimension mismatches early!

2. Dimension Cheat Sheet:
   - Input tokens: [batch_size, seq_len]
   - Embeddings: [batch_size, seq_len, d_model]
   - Attention matrices: [batch_size, num_heads, seq_len, seq_len]
   - FFN hidden layer: [batch_size, seq_len, d_ff]
   - Output logits: [batch_size, seq_len, vocab_size]

3. Common Pitfalls:
   - Forgetting to scale dot products by √d_k
   - Incorrect mask dimensions or application
   - Missing residual connections
   - Wrong order of layer norm and dropout
   - Tensor dimension mismatches in attention
   - Not handling padding properly

4. Performance Considerations:
   - Memory usage scales with sequence length squared
   - Attention computation is O(n²) with sequence length
   - Balance between d_model and num_heads
   - Trade-off between model size and batch size

## Implementation Details

### Embedding and Positional Encoding
This implements the input embedding and positional encoding from Section 3.5 of the paper. Key points:
- Embedding dimension can differ from model dimension (using projection)
- Positional encoding uses sine and cosine functions
- Scale embeddings by √d_model
- Apply dropout to the sum of embeddings and positional encodings

Implementation tips:
- Use `nn.Embedding` for token embeddings
- Store scaling factor as float during initialization
- Remember to expand positional encoding for batch dimension
- Add assertion for input dtype (should be torch.long)

### Transformer Attention
Implements the core attention mechanism from Section 3.2.1. Formula: Attention(Q,K,V) = softmax(QK^T/√d_k)V

Key points:
- Supports both self-attention and cross-attention
- Handles different sequence lengths for encoder/decoder
- Scales dot products by 1/√d_k
- Applies attention masking before softmax

Implementation tips:
- Use separate Q,K,V projections
- Handle masking through addition (not masked_fill)
- Remember to reshape for multi-head attention
- Keep track of tensor dimensions at each step

### Feed-Forward Network (FFN)
Implements the position-wise feed-forward network from Section 3.3: FFN(x) = max(0, xW₁ + b₁)W₂ + b₂

Key points:
- Two linear transformations with ReLU in between
- Inner layer dimension (d_ff) is typically 2048
- Applied identically to each position

Implementation tips:
- Use nn.Linear for transformations
- Remember to include bias terms
- Position-wise means same transformation for each position
- Dimension flow: d_model → d_ff → d_model

### Transformer Decoder
Implements decoder layer from Section 3.1, with three sub-layers:
- Masked multi-head self-attention
- Multi-head cross-attention with encoder output
- Position-wise feed-forward network

Key points:
- Self-attention uses causal masking
- Cross-attention allows attending to all encoder outputs
- Each sub-layer followed by residual connection and layer normalization

Key implementation detail for causal masking:
- Create causal mask using upper triangular matrix:
 ```python
 mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
 mask = mask.masked_fill(mask == 1, float('-inf'))
 ```

This creates a pattern where position i can only attend to positions ≤ i
Using -inf ensures zero attention to future positions after softmax
Visualization of mask for seq_len=5:\
 [[0, -inf, -inf, -inf, -inf],\
 [0,    0, -inf, -inf, -inf],\
 [0,    0,    0, -inf, -inf],\
 [0,    0,    0,    0, -inf],\
 [0,    0,    0,    0,    0]]


Implementation tips:
- Order of operations matters (masking before softmax)
- Each attention layer has its own projections
- Remember to pass encoder outputs for cross-attention
 Careful with mask dimensions in self and cross attention

### Encoder-Decoder Stack
Implements the full stack of encoder and decoder layers from Section 3.1.
Key points:
- Multiple encoder and decoder layers (typically 6)
- Each encoder output feeds into all decoder layers
- Maintains residual connections throughout the stack

Implementation tips:
- Use nn.ModuleList for layer stacks
- Share encoder outputs across decoder layers
- Maintain consistent masking throughout
- Handle padding masks separately from causal masks

### Full Transformer
Combines all components into complete architecture:
- Input embeddings for source and target
- Positional encoding
- Encoder-decoder stack
- Final linear and softmax layer

Key points:
- Handles different vocabulary sizes for source/target
- Shifts decoder inputs for teacher forcing
- Projects outputs to target vocabulary size
- Applies log softmax for training stability

Implementation tips:
- Handle start tokens for decoder input
- Maintain separate embeddings for source/target
- Remember to scale embeddings
- Consider sharing embedding weights with output layer

### Testing
Our implementation includes comprehensive tests for each component:

- Shape preservation through layers
- Masking effectiveness
- Attention pattern verification
- Forward/backward pass validation
- Parameter and gradient checks

See the notebook for detailed test implementations and results.

### Visualizations
The implementation includes visualizations of:

- Attention patterns
- Positional encodings
- Masking effects
- Layer connectivity

These visualizations help understand the inner workings of the transformer and verify correct implementation.

For detailed code and interactive examples, please refer to the complete implementation notebook.

## PyData Global 2025 Presentation
For those of you who prefer to learn from videos, you can watch my PyData Global 2025 presentation "[I Built a Transformer from Scratch So You Don’t Have To](https://www.youtube.com/watch?v=ID5zSzycQBg)"

• How the original Transformer architecture works

• How to translate each component into PyTorch

• Key ideas: attention, masking, positional encoding, FFN

• A decoder-only forward pass, step-by-step

• Common implementation bugs — and how to debug them

• Where to go next (code, tutorials, training references)




## Related Implementations

This repository is part of a series implementing the key architectural innovations from the DeepSeek paper:

1. **[Transformer Implementation Tutorial](https://huggingface.co/datasets/bird-of-paradise/transformer-from-scratch-tutorial)**(This Tutorial): A detailed tutorial on implementing transformer architecture with explanations of key components.

2. **[DeepSeek Multi-head Latent Attention](https://huggingface.co/bird-of-paradise/deepseek-mla)**: Implementation of DeepSeek's MLA mechanism for efficient KV cache usage during inference.

3. **[DeepSeek MoE](https://huggingface.co/bird-of-paradise/deepseek-moe)**: Implementation of DeepSeek's Mixture of Experts architecture that enables efficient scaling of model parameters.

Together, these implementations cover the core innovations that power DeepSeek's state-of-the-art performance. By combining the MoE architecture with Multi-head Latent Attention, you can build a complete DeepSeek-style model with improved training efficiency and inference performance.
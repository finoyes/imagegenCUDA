# Transformer Image Generator — Complete File Reference

Every `.cu`, `.cuh`, `.cpp`, `.h` file explained: the concept, the math, and how to code it.

---

# PART 1 — CUDA KERNELS (`cuda/kernels/`)

These are the atomic GPU operations. Every higher-level file calls into these.
Understand these first — everything else is orchestration on top.

---

## `matmul.cuh` / `matmul.cu`

### Concept

Matrix multiplication is the single most-executed operation in a transformer.
Every linear projection (Q, K, V, output, FFN) is a GEMM:

```
C = A × B      where A is (M×K), B is (K×N), C is (M×N)
```

A naive implementation assigns one thread per output element and loops over K.
That's correct but slow — every thread re-reads the same rows/columns from global
memory (DRAM), which has ~500 cycle latency.

**Tiled GEMM** fixes this by loading a `TILE×TILE` block of A and B into shared
memory (L1-speed, ~5 cycle latency), doing the partial dot-products entirely in
shared memory, then sliding the tile across K. Each element of global memory is
read once per tile instead of once per output element.

### Math

For tile size T, the algorithm computes:

```
C[row][col] = Σ_t  (A_tile[row][0..T] · B_tile[0..T][col])
```

iterated across `ceil(K/T)` tiles.

### Header — `matmul.cuh`

```cpp
#pragma once
#include <cuda_runtime.h>

// C = alpha * A * B + beta * C
// A: (M x K), B: (K x N), C: (M x N) — all row-major, device pointers
void matmul(
    const float* A, const float* B, float* C,
    int M, int K, int N,
    float alpha = 1.0f, float beta = 0.0f
);

// Batched version: runs matmul independently for each item in a batch
// A: (batch x M x K), B: (batch x K x N), C: (batch x M x N)
void batched_matmul(
    const float* A, const float* B, float* C,
    int batch, int M, int K, int N
);
```

### Implementation — `matmul.cu`

```cpp
#include "matmul.cuh"
#include <cuda_runtime.h>

#define TILE 16   // 16x16 threads per block = 256 threads, fits well in a warp

__global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* C,
    int M, int K, int N,
    float alpha, float beta)
{
    // Each block computes one TILE x TILE sub-matrix of C
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;  // row index in C
    int col = blockIdx.x * TILE + threadIdx.x;  // col index in C

    float acc = 0.0f;

    // Loop over tiles along the K dimension
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        // Load one element of A into shared memory (guard against out-of-bounds)
        if (row < M && t * TILE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load one element of B into shared memory
        if (col < N && t * TILE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();  // wait until the full tile is loaded

        // Dot product within this tile
        for (int k = 0; k < TILE; k++)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();  // wait before overwriting shared memory in next iteration
    }

    // Write result
    if (row < M && col < N) {
        if (beta == 0.0f)
            C[row * N + col] = alpha * acc;
        else
            C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}

void matmul(const float* A, const float* B, float* C,
            int M, int K, int N, float alpha, float beta)
{
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_kernel<<<grid, block>>>(A, B, C, M, K, N, alpha, beta);
}

// Batched: stride through each batch item
__global__ void batched_matmul_kernel(
    const float* A, const float* B, float* C,
    int M, int K, int N)
{
    int b = blockIdx.z;
    // Reuse the same tiled logic, just offset pointers by batch stride
    const float* Ab = A + b * M * K;
    const float* Bb = B + b * K * N;
    float*       Cb = C + b * M * N;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        As[threadIdx.y][threadIdx.x] = (row < M && t*TILE+threadIdx.x < K)
            ? Ab[row * K + t * TILE + threadIdx.x] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (col < N && t*TILE+threadIdx.y < K)
            ? Bb[(t * TILE + threadIdx.y) * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; k++) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) Cb[row * N + col] = acc;
}

void batched_matmul(const float* A, const float* B, float* C,
                    int batch, int M, int K, int N)
{
    dim3 block(TILE, TILE);
    dim3 grid((N+TILE-1)/TILE, (M+TILE-1)/TILE, batch);
    batched_matmul_kernel<<<grid, block>>>(A, B, C, M, K, N);
}
```

**Key concepts to understand:**
- `__shared__` — declares shared memory (per-block scratchpad, fast)
- `__syncthreads()` — barrier: all threads in the block must reach this before any continue
- `blockIdx` / `threadIdx` — 3D coordinates identifying which block/thread is running
- `__restrict__` — tells the compiler A and B don't alias C (enables better optimization)

---

## `softmax.cuh` / `softmax.cu`

### Concept

Softmax converts a vector of raw scores into a probability distribution:

```
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
```

The problem: if any `x_i` is large (e.g. 100), `exp(100)` overflows to infinity.

**The safe version (log-sum-exp trick):**

```
softmax(x_i) = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
```

Subtracting the max doesn't change the output (the max cancels in numerator and
denominator) but keeps all exponents ≤ 0, so they stay in `[0, 1]`.

In CUDA this requires two parallel reductions over the input vector:
1. Find `max(x)` — parallel max reduction
2. Compute `Σ exp(x_i - max)` — parallel sum reduction
Then a final pass divides each element.

### Header — `softmax.cuh`

```cpp
#pragma once
#include <cuda_runtime.h>

// In-place row-wise softmax
// input: (rows x cols) row-major, device pointer
// Each row is softmaxed independently
void softmax(float* input, int rows, int cols);

// Used inside attention: softmax over last dim of (batch x heads x seq x seq)
void attention_softmax(float* scores, int batch, int heads, int seq_len);
```

### Implementation — `softmax.cu`

```cpp
#include "softmax.cuh"

// One block per row. Threads cooperate to find max then sum.
// Works for cols up to blockDim.x * 2 (use 256 threads = up to 512 cols)
// For longer sequences, you'd need a multi-pass reduction.
__global__ void softmax_kernel(float* x, int cols) {
    extern __shared__ float sdata[];  // dynamically sized shared memory

    int row = blockIdx.x;
    float* row_ptr = x + row * cols;

    // Step 1: find max across the row
    float thread_max = -1e38f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        thread_max = fmaxf(thread_max, row_ptr[i]);

    sdata[threadIdx.x] = thread_max;
    __syncthreads();

    // Parallel reduction to find block-wide max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
        __syncthreads();
    }
    float row_max = sdata[0];

    // Step 2: compute exp(x - max) and accumulate sum
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = expf(row_ptr[i] - row_max);
        row_ptr[i] = val;       // overwrite with exp value
        thread_sum += val;
    }

    sdata[threadIdx.x] = thread_sum;
    __syncthreads();

    // Parallel reduction to find total sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();
    }
    float row_sum = sdata[0];

    // Step 3: normalize
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        row_ptr[i] /= row_sum;
}

void softmax(float* input, int rows, int cols) {
    int threads = 256;
    size_t shared = threads * sizeof(float);
    softmax_kernel<<<rows, threads, shared>>>(input, cols);
}

void attention_softmax(float* scores, int batch, int heads, int seq_len) {
    // Treat each (seq_len) row of the (seq x seq) attention matrix as one softmax
    int rows = batch * heads * seq_len;
    softmax(scores, rows, seq_len);
}
```

**Key concepts:**
- `extern __shared__` — shared memory whose size is specified at launch time (the 3rd kernel argument)
- Parallel reduction — a tree-structured operation that halves the active threads each step to compute a sum/max in O(log N) steps instead of O(N)

---

## `layernorm.cuh` / `layernorm.cu`

### Concept

Layer normalization normalizes the activations within each token's feature vector
(as opposed to batch norm which normalizes across the batch dimension):

```
y = (x - mean(x)) / sqrt(var(x) + ε)  * γ + β
```

Where `γ` (gamma) and `β` (beta) are learned per-feature scale and shift parameters.

In a transformer, this is applied before attention and before the FFN
("Pre-LN" style, which is more stable than the original post-LN).

The "fused" kernel computes mean, variance, normalization, and the γ/β affine
transform all in one pass over the data — saving two extra global memory
round-trips compared to doing them as separate kernels.

### Header — `layernorm.cuh`

```cpp
#pragma once
#include <cuda_runtime.h>

// Forward pass — in-place normalization
// input/output: (N x d_model) — each of the N tokens gets normalized independently
// gamma, beta: (d_model,) — learned parameters
// eps: numerical stability constant (typically 1e-5)
void layernorm_forward(
    float* output,
    const float* input,
    const float* gamma,
    const float* beta,
    int N,          // number of tokens (batch_size * seq_len)
    int d_model,
    float eps = 1e-5f
);

// Backward pass — computes gradients w.r.t. input, gamma, beta
void layernorm_backward(
    float* d_input,          // gradient to propagate back
    float* d_gamma,          // gradient for gamma (accumulated)
    float* d_beta,           // gradient for beta (accumulated)
    const float* d_output,   // gradient from next layer
    const float* input,      // saved from forward pass
    const float* gamma,
    int N,
    int d_model,
    float eps = 1e-5f
);
```

### Implementation — `layernorm.cu`

```cpp
#include "layernorm.cuh"
#include <cuda_runtime.h>

// One block per token. Threads cooperate over d_model features.
__global__ void layernorm_fwd_kernel(
    float* out, const float* in,
    const float* gamma, const float* beta,
    int d_model, float eps)
{
    extern __shared__ float sdata[];

    int token = blockIdx.x;              // which token
    const float* x = in  + token * d_model;
    float*       y = out + token * d_model;

    // Step 1: compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < d_model; i += blockDim.x)
        sum += x[i];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float mean = sdata[0] / d_model;

    // Step 2: compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        float diff = x[i] - mean;
        var_sum += diff * diff;
    }
    sdata[threadIdx.x] = var_sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float inv_std = rsqrtf(sdata[0] / d_model + eps);  // 1 / sqrt(var + eps)

    // Step 3: normalize and apply affine transform
    for (int i = threadIdx.x; i < d_model; i += blockDim.x)
        y[i] = gamma[i] * ((x[i] - mean) * inv_std) + beta[i];
}

void layernorm_forward(float* output, const float* input,
                       const float* gamma, const float* beta,
                       int N, int d_model, float eps)
{
    int threads = min(256, d_model);
    size_t shared = threads * sizeof(float);
    layernorm_fwd_kernel<<<N, threads, shared>>>(
        output, input, gamma, beta, d_model, eps);
}

// Backward pass — see "Layer Normalization" paper (Ba et al. 2016) for derivation
__global__ void layernorm_bwd_kernel(
    float* dx, float* dgamma, float* dbeta,
    const float* dy, const float* x, const float* gamma,
    int d_model, float eps)
{
    extern __shared__ float sdata[];
    int token = blockIdx.x;
    const float* xi  = x  + token * d_model;
    const float* dyi = dy + token * d_model;
    float*       dxi = dx + token * d_model;

    // Recompute mean and inv_std (or you could save them in the forward pass)
    float sum = 0.0f;
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) sum += xi[i];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float mean = sdata[0] / d_model;

    float vsum = 0.0f;
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        float d = xi[i] - mean; vsum += d * d;
    }
    sdata[threadIdx.x] = vsum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float inv_std = rsqrtf(sdata[0] / d_model + eps);

    // Gradients: dL/dbeta = sum(dy), dL/dgamma = sum(dy * x_hat)
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        float x_hat = (xi[i] - mean) * inv_std;
        atomicAdd(&dbeta[i],  dyi[i]);
        atomicAdd(&dgamma[i], dyi[i] * x_hat);
    }

    // dL/dx — see the paper for the full chain rule derivation
    // Simplified: dx = (1/std) * (dy - mean(dy) - x_hat * mean(dy * x_hat)) * gamma
    float sum_dy = 0.0f, sum_dy_xhat = 0.0f;
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        float x_hat = (xi[i] - mean) * inv_std;
        sum_dy      += dyi[i] * gamma[i];
        sum_dy_xhat += dyi[i] * gamma[i] * x_hat;
    }
    // reduce sum_dy and sum_dy_xhat — using sdata twice
    sdata[threadIdx.x] = sum_dy;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float mean_dy = sdata[0] / d_model;

    sdata[threadIdx.x] = sum_dy_xhat;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float mean_dy_xhat = sdata[0] / d_model;

    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        float x_hat = (xi[i] - mean) * inv_std;
        dxi[i] = inv_std * (dyi[i] * gamma[i] - mean_dy - x_hat * mean_dy_xhat);
    }
}

void layernorm_backward(float* d_input, float* d_gamma, float* d_beta,
                        const float* d_output, const float* input,
                        const float* gamma, int N, int d_model, float eps)
{
    int threads = min(256, d_model);
    size_t shared = threads * sizeof(float);
    layernorm_bwd_kernel<<<N, threads, shared>>>(
        d_input, d_gamma, d_beta, d_output, input, gamma, d_model, eps);
}
```

---

## `activations.cuh` / `activations.cu`

### Concept

Activations introduce non-linearity after linear projections. Without them,
stacking linear layers is still just one linear transformation.

**GELU (Gaussian Error Linear Unit)** is what transformers use in the FFN:

```
GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
```

It's a smooth approximation to ReLU that passes small negative values
instead of zeroing them — this helps gradient flow.

**GELU backward:**

```
GELU'(x) = 0.5 * tanh(t) + (0.5*x + 0.0535161*x²) * sech²(t) + 0.5
where t = sqrt(2/π) * (x + 0.044715*x³)
```

### Header — `activations.cuh`

```cpp
#pragma once
#include <cuda_runtime.h>

void gelu_forward(float* output, const float* input, int n);
void gelu_backward(float* d_input, const float* d_output, const float* input, int n);
void relu_forward(float* output, const float* input, int n);
void relu_backward(float* d_input, const float* d_output, const float* input, int n);
```

### Implementation — `activations.cu`

```cpp
#include "activations.cuh"
#include <cuda_runtime.h>
#include <math.h>

#define GELU_COEF  0.044715f
#define SQRT_2_PI  0.7978845608f   // sqrt(2/pi)

__device__ inline float gelu_val(float x) {
    float t = SQRT_2_PI * (x + GELU_COEF * x * x * x);
    return 0.5f * x * (1.0f + tanhf(t));
}

__device__ inline float gelu_deriv(float x) {
    float t     = SQRT_2_PI * (x + GELU_COEF * x * x * x);
    float tanh_t = tanhf(t);
    float sech2  = 1.0f - tanh_t * tanh_t;
    float dt_dx  = SQRT_2_PI * (1.0f + 3.0f * GELU_COEF * x * x);
    return 0.5f * (1.0f + tanh_t) + 0.5f * x * sech2 * dt_dx;
}

__global__ void gelu_fwd_kernel(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = gelu_val(in[i]);
}

__global__ void gelu_bwd_kernel(float* din, const float* dout, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) din[i] = dout[i] * gelu_deriv(in[i]);
}

__global__ void relu_fwd_kernel(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fmaxf(0.0f, in[i]);
}

__global__ void relu_bwd_kernel(float* din, const float* dout, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) din[i] = (in[i] > 0.0f) ? dout[i] : 0.0f;
}

void gelu_forward(float* output, const float* input, int n) {
    int threads = 256;
    gelu_fwd_kernel<<<(n + threads - 1) / threads, threads>>>(output, input, n);
}
void gelu_backward(float* d_input, const float* d_output, const float* input, int n) {
    int threads = 256;
    gelu_bwd_kernel<<<(n + threads - 1) / threads, threads>>>(d_input, d_output, input, n);
}
void relu_forward(float* output, const float* input, int n) {
    int threads = 256;
    relu_fwd_kernel<<<(n + threads - 1) / threads, threads>>>(output, input, n);
}
void relu_backward(float* d_input, const float* d_output, const float* input, int n) {
    int threads = 256;
    relu_bwd_kernel<<<(n + threads - 1) / threads, threads>>>(d_input, d_output, input, n);
}
```

---

## `attention.cuh` / `attention.cu`

### Concept

This is the core operation of a transformer. Multi-head self-attention lets
every token attend to every other token and aggregate information.

**Single-head attention:**

```
Q = X · W_Q    (queries)
K = X · W_K    (keys)
V = X · W_V    (values)

Attention(Q,K,V) = softmax(Q·Kᵀ / √d_k) · V
```

The `Q·Kᵀ` produces a `(seq_len x seq_len)` score matrix — how much each
token should attend to every other token. Dividing by `√d_k` prevents the
dot products from growing too large and pushing softmax into near-zero gradients.

**Multi-head:** Run H independent attention heads in parallel, each with their
own `W_Q`, `W_K`, `W_V` projections of size `d_k = d_model / H`. Concatenate
the H outputs and project through `W_O`.

**Forward pass steps:**
1. Project: compute Q, K, V matrices via matmul
2. Split into H heads (just a reshape + stride)
3. Score: compute `QKᵀ / √d_k` per head
4. Softmax over sequence dimension
5. Weighted sum: multiply scores by V
6. Concatenate heads
7. Output projection: multiply by W_O

### Header — `attention.cuh`

```cpp
#pragma once
#include <cuda_runtime.h>

struct AttentionParams {
    // Weight matrices (device pointers) — shapes noted as comments
    float* W_Q;   // (d_model x d_model)
    float* W_K;   // (d_model x d_model)
    float* W_V;   // (d_model x d_model)
    float* W_O;   // (d_model x d_model)
    float* b_Q;   // (d_model,)  biases
    float* b_K;
    float* b_V;
    float* b_O;

    // Gradients (same shapes, allocated alongside weights)
    float* dW_Q; float* dW_K; float* dW_V; float* dW_O;
    float* db_Q; float* db_K; float* db_V; float* db_O;
};

struct AttentionCache {
    // Saved tensors needed for backward pass
    float* Q;         // (batch x seq x d_model)
    float* K;
    float* V;
    float* scores;    // (batch x heads x seq x seq) — post-softmax
    float* input;     // reference to input, not owned
};

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
```

### Implementation — `attention.cu`

```cpp
#include "attention.cuh"
#include "matmul.cuh"
#include "softmax.cuh"
#include <cuda_runtime.h>
#include <math.h>

// Add bias to every row: out[i][j] += bias[j]
__global__ void add_bias_kernel(float* out, const float* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols)
        out[idx] += bias[idx % cols];
}

// Scale: multiply every element by a scalar
__global__ void scale_kernel(float* x, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= scale;
}

// Transpose last two dims: (batch x heads x seq x d_k) -> (batch x heads x d_k x seq)
__global__ void transpose_kernel(float* out, const float* in,
                                  int batch, int heads, int seq, int dk) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < seq * dk) {
        int s = i / dk, k = i % dk;
        out[b*heads*seq*dk + h*seq*dk + s*dk + k] =
            in [b*heads*dk*seq + h*dk*seq + k*seq + s];
    }
}

void attention_forward(
    float* output, AttentionCache* cache,
    const float* input, const AttentionParams* p,
    int batch, int seq_len, int d_model, int num_heads)
{
    int d_k    = d_model / num_heads;
    int total  = batch * seq_len;          // treat batch*seq as the "rows"
    int threads = 256;

    // ---- Step 1: Linear projections Q, K, V ----
    // input: (total x d_model), W_Q: (d_model x d_model) -> Q: (total x d_model)
    matmul(input, p->W_Q, cache->Q, total, d_model, d_model);
    matmul(input, p->W_K, cache->K, total, d_model, d_model);
    matmul(input, p->W_V, cache->V, total, d_model, d_model);

    int n_qkv = total * d_model;
    add_bias_kernel<<<(n_qkv+threads-1)/threads, threads>>>(cache->Q, p->b_Q, total, d_model);
    add_bias_kernel<<<(n_qkv+threads-1)/threads, threads>>>(cache->K, p->b_K, total, d_model);
    add_bias_kernel<<<(n_qkv+threads-1)/threads, threads>>>(cache->V, p->b_V, total, d_model);

    // ---- Step 2: Compute attention scores per head ----
    // Reshape Q,K to (batch x heads x seq x d_k) — this is just a reinterpretation
    // since memory is already laid out correctly if d_model = heads * d_k
    // scores: (batch x heads x seq x seq)
    float scale = 1.0f / sqrtf((float)d_k);

    // For each batch item and each head, compute Q_h * K_h^T
    // Using batched matmul: treat (batch*heads) as the batch dimension
    // Q_h: (seq x d_k), K_h^T: (d_k x seq) -> scores: (seq x seq)
    batched_matmul(
        cache->Q, cache->K, cache->scores,
        batch * num_heads, seq_len, d_k, seq_len
    );

    // Scale scores by 1/sqrt(d_k)
    int n_scores = batch * num_heads * seq_len * seq_len;
    scale_kernel<<<(n_scores+threads-1)/threads, threads>>>(cache->scores, scale, n_scores);

    // ---- Step 3: Softmax over last dim (seq) ----
    attention_softmax(cache->scores, batch, num_heads, seq_len);

    // ---- Step 4: Weighted sum of values ----
    // scores: (batch*heads x seq x seq), V: (batch*heads x seq x d_k)
    // -> out_heads: (batch*heads x seq x d_k) = (batch x seq x d_model)
    batched_matmul(
        cache->scores, cache->V, output,
        batch * num_heads, seq_len, seq_len, d_k
    );

    // ---- Step 5: Output projection ----
    // output is now (total x d_model), project through W_O
    // (do this in-place using a temp buffer in practice)
    matmul(output, p->W_O, output, total, d_model, d_model);
    add_bias_kernel<<<(n_qkv+threads-1)/threads, threads>>>(output, p->b_O, total, d_model);
}

// Backward is the chain rule through each step above in reverse order.
// This is the most math-intensive part — see "Attention Is All You Need" appendix
// and Andrej Karpathy's makemore/nanoGPT for worked-out gradient derivations.
void attention_backward(
    float* d_input, AttentionParams* p,
    const float* d_output, const AttentionCache* cache,
    int batch, int seq_len, int d_model, int num_heads)
{
    int d_k   = d_model / num_heads;
    int total = batch * seq_len;

    // Step 5 backward: d_output → dW_O, db_O, d_pre_output
    // (transpose matmul gradient logic)
    matmul(d_output, p->W_O, /* d_pre_output */ nullptr, total, d_model, d_model);
    // dW_O += output^T * d_output  (accumulate)

    // Step 4 backward: d_pre_output → d_scores, dV
    // d_scores = d_pre_output * V^T
    // dV       = scores^T * d_pre_output

    // Step 3 backward: softmax backward
    // d_scores_pre_softmax[i] = scores[i] * (d_scores[i] - sum_j(scores[j]*d_scores[j]))

    // Step 2 backward: scale backward (multiply by same scale factor)

    // Step 1 backward: dQ = d_score_scaled * K, dW_Q += input^T * dQ, etc.
    // d_input = dQ * W_Q^T + dK * W_K^T + dV * W_V^T

    // Full implementation follows the same pattern as the forward pass
    // but with transposed matmuls — highly recommended to derive on paper first
}
```

---

# PART 2 — MODEL (`cuda/model/`)

These files compose the kernels into the transformer architecture.

---

## `encoder.cuh` / `encoder.cu`

### Concept

The encoder converts a raw image into a sequence of token embeddings that the
transformer can process. It does two things:

**1. Patch embedding**

Split the image into a grid of non-overlapping patches (e.g. a 64×64 image with
8×8 patches gives 64 patches). Flatten each patch into a vector of size
`patch_size² × channels` (8×8×3 = 192 for RGB). Project that vector to `d_model`
dimensions via a learned linear layer. This is conceptually identical to the
word embedding in a language transformer, just applied to image patches.

**2. Positional encoding**

Transformers have no inherent sense of order — attention treats inputs as a
set. Positional encodings inject position information by adding a
position-dependent vector to each token embedding.

Two options:
- **Sinusoidal (fixed):** `PE[pos][2i] = sin(pos / 10000^(2i/d_model))`,
  `PE[pos][2i+1] = cos(...)`. No parameters, generalizes to unseen lengths.
- **Learned:** Each position has a learnable embedding vector. Simpler to
  implement but fixed to the training sequence length. Used in ViT.

### Header — `encoder.cuh`

```cpp
#pragma once
#include <cuda_runtime.h>

struct EncoderParams {
    float* patch_proj_W;    // (patch_dim x d_model) — patch_dim = P*P*C
    float* patch_proj_b;    // (d_model,)
    float* pos_embedding;   // (num_patches x d_model) — learned positional embeddings
    float* cls_token;       // (d_model,) — optional CLS token (like ViT)

    float* d_patch_proj_W;
    float* d_patch_proj_b;
    float* d_pos_embedding;
};

struct EncoderConfig {
    int image_size;      // e.g. 64
    int patch_size;      // e.g. 8
    int channels;        // e.g. 3 (RGB)
    int d_model;         // e.g. 256
    int num_patches;     // (image_size / patch_size)^2
    int patch_dim;       // patch_size * patch_size * channels
};

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
```

### Implementation — `encoder.cu`

```cpp
#include "encoder.cuh"
#include "matmul.cuh"
#include <cuda_runtime.h>
#include <math.h>

// Extract patches from image and flatten them into rows
// images: (batch x C x H x W)
// patches: (batch x num_patches x patch_dim)
__global__ void extract_patches_kernel(
    float* patches, const float* images,
    int batch, int C, int H, int W,
    int patch_size, int num_patches_h)  // num_patches_h = H/patch_size
{
    // Each thread handles one element of the output
    int b  = blockIdx.z;
    int p  = blockIdx.y;   // patch index
    int d  = blockIdx.x * blockDim.x + threadIdx.x;  // position in flattened patch

    int patch_dim = patch_size * patch_size * C;
    if (d >= patch_dim) return;

    // Patch coordinates in the image grid
    int ph = p / num_patches_h;   // patch row
    int pw = p % num_patches_h;   // patch col

    // Position within the patch
    int c  =  d / (patch_size * patch_size);
    int py = (d % (patch_size * patch_size)) / patch_size;
    int px =  d % patch_size;

    // Source pixel position in the image
    int iy = ph * patch_size + py;
    int ix = pw * patch_size + px;

    patches[b * (num_patches_h*num_patches_h) * patch_dim + p * patch_dim + d]
        = images[b * C * H * W + c * H * W + iy * W + ix];
}

// Add positional embeddings in-place: output[b][p] += pos_emb[p]
__global__ void add_positional_encoding_kernel(
    float* output, const float* pos_emb,
    int batch, int num_patches, int d_model)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * num_patches * d_model) return;
    int p = (idx / d_model) % num_patches;
    output[idx] += pos_emb[p * d_model + idx % d_model];
}

void encoder_forward(
    float* output, const float* images,
    const EncoderParams* params, const EncoderConfig* cfg, int batch)
{
    int P  = cfg->patch_size, C = cfg->channels;
    int H  = cfg->image_size, W = cfg->image_size;
    int NP = cfg->num_patches;
    int PD = cfg->patch_dim;

    // 1. Extract patches — allocate a temp buffer
    float* d_patches;  // (batch x NP x PD)
    cudaMalloc(&d_patches, batch * NP * PD * sizeof(float));

    int NPH = H / P;
    dim3 grid_ext((PD + 255)/256, NP, batch);
    extract_patches_kernel<<<grid_ext, 256>>>(
        d_patches, images, batch, C, H, W, P, NPH);

    // 2. Project patches to d_model: (batch*NP x PD) * (PD x d_model) -> (batch*NP x d_model)
    matmul(d_patches, params->patch_proj_W, output,
           batch * NP, PD, cfg->d_model);

    // 3. Add patch projection bias
    int total = batch * NP * cfg->d_model;
    // (add_bias_kernel from attention.cu — consider putting in a shared utils header)

    // 4. Add positional embeddings
    add_positional_encoding_kernel<<<(total+255)/256, 256>>>(
        output, params->pos_embedding, batch, NP, cfg->d_model);

    cudaFree(d_patches);
}

void encoder_init(EncoderParams* params, const EncoderConfig* cfg) {
    // Kaiming uniform initialization for the patch projection:
    // std = sqrt(2 / patch_dim)
    // Fill on CPU then cudaMemcpy, or use cuRAND for GPU-side initialization

    // Sinusoidal positional encoding (fixed, copy to pos_embedding):
    int NP = cfg->num_patches;
    int D  = cfg->d_model;
    std::vector<float> pos_enc(NP * D);
    for (int p = 0; p < NP; p++) {
        for (int i = 0; i < D; i += 2) {
            float freq = 1.0f / powf(10000.0f, (float)i / D);
            pos_enc[p * D + i]     = sinf(p * freq);
            pos_enc[p * D + i + 1] = cosf(p * freq);
        }
    }
    cudaMemcpy(params->pos_embedding, pos_enc.data(),
               NP * D * sizeof(float), cudaMemcpyHostToDevice);
}
```

---

## `decoder.cuh` / `decoder.cu`

### Concept

The decoder reconstructs the output image from the transformer's token sequence.
For a simple autoencoder-style generator, this is just reversing the encoder:

**Option A (simple) — linear pixel head:**
Project each output token from `d_model` back to `patch_dim` = `P×P×C`,
then scatter the patches back into their image positions.

**Option B (better) — progressive upsampling:**
Project to a small spatial feature map, then apply a series of
transposed convolutions (upconv) to double resolution at each step until
you reach the target image size. This produces sharper images than the
linear head.

For a first implementation, start with Option A.

### Header — `decoder.cuh`

```cpp
#pragma once
#include <cuda_runtime.h>
#include "encoder.cuh"   // for EncoderConfig

struct DecoderParams {
    float* proj_W;   // (d_model x patch_dim) — inverse of encoder projection
    float* proj_b;   // (patch_dim,)
    float* d_proj_W;
    float* d_proj_b;
};

// Forward: token sequence -> reconstructed image
// input:  (batch x num_patches x d_model)
// output: (batch x C x H x W)
void decoder_forward(
    float* output,
    const float* input,
    const DecoderParams* params,
    const EncoderConfig* cfg,
    int batch
);

void decoder_backward(
    float* d_input,
    DecoderParams* params,
    const float* d_output,
    const float* input,
    const EncoderConfig* cfg,
    int batch
);
```

### Implementation — `decoder.cu`

```cpp
#include "decoder.cuh"
#include "matmul.cuh"
#include <cuda_runtime.h>

// Reassemble patches back into an image (inverse of extract_patches_kernel)
__global__ void reassemble_patches_kernel(
    float* image, const float* patches,
    int batch, int C, int H, int W,
    int patch_size, int num_patches_h)
{
    int b = blockIdx.z, p = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int patch_dim = patch_size * patch_size * C;
    if (d >= patch_dim) return;

    int ph = p / num_patches_h, pw = p % num_patches_h;
    int c  =  d / (patch_size * patch_size);
    int py = (d % (patch_size * patch_size)) / patch_size;
    int px =  d % patch_size;
    int iy = ph * patch_size + py, ix = pw * patch_size + px;

    image[b*C*H*W + c*H*W + iy*W + ix]
        = patches[b * (num_patches_h*num_patches_h) * patch_dim + p * patch_dim + d];
}

// Clamp output to [-1, 1] or [0, 1] depending on normalization
__global__ void clamp_kernel(float* x, float lo, float hi, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = fmaxf(lo, fminf(hi, x[i]));
}

void decoder_forward(
    float* output, const float* input,
    const DecoderParams* params, const EncoderConfig* cfg, int batch)
{
    int NP = cfg->num_patches, PD = cfg->patch_dim;
    int C = cfg->channels, H = cfg->image_size, W = cfg->image_size;
    int P = cfg->patch_size, NPH = H / P;

    // 1. Project tokens back to patch space
    float* d_patches;
    cudaMalloc(&d_patches, batch * NP * PD * sizeof(float));
    matmul(input, params->proj_W, d_patches, batch * NP, cfg->d_model, PD);
    // add bias

    // 2. Reassemble patches into image
    dim3 grid_re((PD+255)/256, NP, batch);
    reassemble_patches_kernel<<<grid_re, 256>>>(
        output, d_patches, batch, C, H, W, P, NPH);

    // 3. Clamp to valid pixel range
    int total = batch * C * H * W;
    clamp_kernel<<<(total+255)/256, 256>>>(output, -1.0f, 1.0f, total);

    cudaFree(d_patches);
}
```

---

## `transformer.cuh` / `transformer.cu`

### Concept

This is the full transformer block stack. Each block contains:

```
x = x + Attention(LayerNorm(x))       // self-attention with residual
x = x + FFN(LayerNorm(x))             // feed-forward with residual
```

The FFN is a 2-layer MLP:

```
FFN(x) = GELU(x · W1 + b1) · W2 + b2
         where W1: (d_model x d_ff), W2: (d_ff x d_model)
         and d_ff is typically 4 * d_model
```

Residual connections (the `x + ...`) are critical for training deep networks —
they give gradients a direct path back through the network and prevent the
vanishing gradient problem.

### Header — `transformer.cuh`

```cpp
#pragma once
#include <cuda_runtime.h>
#include "attention.cuh"
#include "layernorm.cuh"

struct FFNParams {
    float* W1; float* b1;   // (d_model x d_ff), (d_ff,)
    float* W2; float* b2;   // (d_ff x d_model), (d_model,)
    float* dW1; float* db1;
    float* dW2; float* db2;
};

struct TransformerBlockParams {
    AttentionParams attn;
    FFNParams ffn;
    // Layer norm parameters (2 norms per block)
    float* ln1_gamma; float* ln1_beta;
    float* ln2_gamma; float* ln2_beta;
    float* d_ln1_gamma; float* d_ln1_beta;
    float* d_ln2_gamma; float* d_ln2_beta;
};

struct TransformerParams {
    TransformerBlockParams* blocks;   // array of num_layers blocks
    int num_layers;
    int d_model;
    int num_heads;
    int d_ff;
};

struct TransformerCache {
    // Per-block intermediate activations needed for backward pass
    float** ln1_out;      // output of first layer norm  [num_layers]
    float** attn_out;     // output of attention         [num_layers]
    float** ln2_out;      // output of second layer norm [num_layers]
    float** ffn_mid;      // FFN hidden activations       [num_layers]
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
```

### Implementation — `transformer.cu`

```cpp
#include "transformer.cuh"
#include "../kernels/attention.cuh"
#include "../kernels/layernorm.cuh"
#include "../kernels/activations.cuh"
#include "../kernels/matmul.cuh"
#include <cuda_runtime.h>
#include <cstring>

// Element-wise add: out += residual
__global__ void add_residual_kernel(float* out, const float* residual, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += residual[i];
}

void transformer_forward(
    float* output, TransformerCache* cache,
    const float* input, const TransformerParams* p,
    int batch, int seq_len)
{
    int D   = p->d_model;
    int N   = batch * seq_len;   // total tokens
    int dff = p->d_ff;

    // Copy input to output buffer (we'll modify in-place)
    cudaMemcpy(output, input, N * D * sizeof(float), cudaMemcpyDeviceToDevice);

    for (int l = 0; l < p->num_layers; l++) {
        TransformerBlockParams& bp = p->blocks[l];

        // ---- Sub-layer 1: Pre-LN → Attention → Residual ----

        // Pre-LayerNorm
        layernorm_forward(cache->ln1_out[l], output,
                          bp.ln1_gamma, bp.ln1_beta, N, D);

        // Self-attention
        attention_forward(cache->attn_out[l], &cache->attn_caches[l],
                          cache->ln1_out[l], &bp.attn,
                          batch, seq_len, D, p->num_heads);

        // Residual: output = output + attn_out
        add_residual_kernel<<<(N*D+255)/256, 256>>>(output, cache->attn_out[l], N*D);

        // ---- Sub-layer 2: Pre-LN → FFN → Residual ----

        // Pre-LayerNorm
        layernorm_forward(cache->ln2_out[l], output,
                          bp.ln2_gamma, bp.ln2_beta, N, D);

        // FFN: Linear → GELU → Linear
        matmul(cache->ln2_out[l], bp.ffn.W1, cache->ffn_mid[l], N, D, dff);
        // add b1
        gelu_forward(cache->ffn_mid[l], cache->ffn_mid[l], N * dff);
        matmul(cache->ffn_mid[l], bp.ffn.W2, /* temp */ output, N, dff, D);
        // NOTE: in practice use a separate temp buffer, not output directly
        // add b2

        // Residual: output = output + ffn_out
        // (already in output if you use a temp buffer correctly)
        add_residual_kernel<<<(N*D+255)/256, 256>>>(output, /* ffn_out */ nullptr, N*D);
    }
}

// Backward: reverse the above in reverse layer order
// For each block, going backwards:
// 1. Backprop through residual (gradient just flows through addition unchanged)
// 2. Backprop through FFN: Linear2 → GELU → Linear1
// 3. Backprop through LayerNorm2
// 4. Backprop through attention
// 5. Backprop through LayerNorm1
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
```

---

# PART 3 — TRAINING (`cuda/training/`)

---

## `loss.cuh` / `loss.cu`

### Concept

For image reconstruction (autoencoder), the training signal is how different
the reconstructed image is from the original.

**Mean Squared Error (MSE):**

```
L = (1/N) Σ (y_pred - y_true)²
dL/dy_pred = (2/N) * (y_pred - y_true)
```

Simple and fast. The gradient points directly at the prediction error.

**For image generation you may also want:**

Perceptual loss: pass both the predicted and target image through a pretrained
feature extractor (e.g. first few layers of VGG) and compare feature maps.
This produces sharper images than pixel-MSE alone. Requires a pretrained feature
extractor — add later once MSE training works.

### Header — `loss.cuh`

```cpp
#pragma once
#include <cuda_runtime.h>

// Computes MSE loss and its gradient
// pred, target: (n,) device pointers
// loss_out: scalar output (single float, device pointer)
// grad_out: (n,) gradient w.r.t. pred
void mse_loss(
    float* loss_out,
    float* grad_out,
    const float* pred,
    const float* target,
    int n
);
```

### Implementation — `loss.cu`

```cpp
#include "loss.cuh"
#include <cuda_runtime.h>

__global__ void mse_grad_kernel(
    float* grad, const float* pred, const float* target, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad[i] = 2.0f * (pred[i] - target[i]) / n;
}

// Simple sum reduction for the loss value
__global__ void mse_loss_kernel(
    float* loss, const float* pred, const float* target, int n)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;

    float val = 0.0f;
    if (i < n) {
        float diff = pred[i] - target[i];
        val = diff * diff;
    }
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(loss, sdata[0] / n);  // accumulate from all blocks
}

void mse_loss(float* loss_out, float* grad_out,
              const float* pred, const float* target, int n)
{
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;

    // Reset loss accumulator
    cudaMemset(loss_out, 0, sizeof(float));

    mse_loss_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        loss_out, pred, target, n);
    mse_grad_kernel<<<blocks, threads>>>(
        grad_out, pred, target, n);
}
```

---

## `optimizer.cuh` / `optimizer.cu`

### Concept

**AdamW** (Adam with decoupled weight decay) is the standard optimizer for
transformers. It maintains a running estimate of the gradient mean (m) and
uncentered variance (v) for each parameter, and uses these to produce
adaptive per-parameter learning rates.

Update rule for each parameter `θ`:

```
g_t  = gradient at step t
m_t  = β₁ * m_{t-1} + (1 - β₁) * g_t           (first moment — gradient mean)
v_t  = β₂ * v_{t-1} + (1 - β₂) * g_t²          (second moment — gradient variance)
m̂_t  = m_t / (1 - β₁ᵗ)                         (bias correction)
v̂_t  = v_t / (1 - β₂ᵗ)                         (bias correction)
θ_t  = θ_{t-1} * (1 - lr * λ)  -  lr * m̂_t / (√v̂_t + ε)
```

The `(1 - lr * λ)` term is the **weight decay** applied directly to the
parameter — this is what makes it AdamW vs Adam (in plain Adam, weight decay
is mixed into the gradient, which interacts badly with the adaptive learning
rate scaling).

Standard values: `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`, `λ = 0.01`.

### Header — `optimizer.cuh`

```cpp
#pragma once
#include <cuda_runtime.h>
#include <vector>

struct AdamWState {
    float* m;          // first moment buffer  (same size as params)
    float* v;          // second moment buffer
    int    n;          // number of parameters in this tensor
};

struct AdamWOptimizer {
    std::vector<float*>      params;    // device pointers to weight tensors
    std::vector<float*>      grads;     // device pointers to gradient tensors
    std::vector<AdamWState>  states;    // m, v buffers per tensor
    std::vector<int>         sizes;     // element count per tensor

    float lr;      // learning rate
    float beta1;   // 0.9
    float beta2;   // 0.999
    float eps;     // 1e-8
    float weight_decay;  // 0.01
    int   step;    // current step (for bias correction)
};

AdamWOptimizer adamw_create(float lr, float beta1 = 0.9f,
                             float beta2 = 0.999f, float eps = 1e-8f,
                             float weight_decay = 0.01f);

void adamw_add_param(AdamWOptimizer* opt, float* param, float* grad, int n);
void adamw_step(AdamWOptimizer* opt);
void adamw_zero_grad(AdamWOptimizer* opt);
void adamw_destroy(AdamWOptimizer* opt);
```

### Implementation — `optimizer.cu`

```cpp
#include "optimizer.cuh"
#include <cuda_runtime.h>
#include <math.h>

__global__ void adamw_update_kernel(
    float* param, float* grad, float* m, float* v,
    float lr, float beta1, float beta2, float eps, float wd,
    float bias_corr1, float bias_corr2,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float g = grad[i];

    // Update moments
    m[i] = beta1 * m[i] + (1.0f - beta1) * g;
    v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;

    // Bias-corrected moments
    float m_hat = m[i] / bias_corr1;
    float v_hat = v[i] / bias_corr2;

    // Weight decay + Adam update
    param[i] = param[i] * (1.0f - lr * wd)
               - lr * m_hat / (sqrtf(v_hat) + eps);
}

AdamWOptimizer adamw_create(float lr, float beta1, float beta2,
                             float eps, float weight_decay)
{
    AdamWOptimizer opt;
    opt.lr = lr; opt.beta1 = beta1; opt.beta2 = beta2;
    opt.eps = eps; opt.weight_decay = weight_decay;
    opt.step = 0;
    return opt;
}

void adamw_add_param(AdamWOptimizer* opt, float* param, float* grad, int n) {
    AdamWState state;
    state.n = n;
    cudaMalloc(&state.m, n * sizeof(float));
    cudaMalloc(&state.v, n * sizeof(float));
    cudaMemset(state.m, 0, n * sizeof(float));
    cudaMemset(state.v, 0, n * sizeof(float));
    opt->params.push_back(param);
    opt->grads.push_back(grad);
    opt->states.push_back(state);
    opt->sizes.push_back(n);
}

void adamw_step(AdamWOptimizer* opt) {
    opt->step++;
    float bc1 = 1.0f - powf(opt->beta1, opt->step);
    float bc2 = 1.0f - powf(opt->beta2, opt->step);

    int threads = 256;
    for (int i = 0; i < (int)opt->params.size(); i++) {
        int n = opt->sizes[i];
        adamw_update_kernel<<<(n+threads-1)/threads, threads>>>(
            opt->params[i], opt->grads[i],
            opt->states[i].m, opt->states[i].v,
            opt->lr, opt->beta1, opt->beta2, opt->eps, opt->weight_decay,
            bc1, bc2, n);
    }
}

void adamw_zero_grad(AdamWOptimizer* opt) {
    for (int i = 0; i < (int)opt->grads.size(); i++)
        cudaMemset(opt->grads[i], 0, opt->sizes[i] * sizeof(float));
}

void adamw_destroy(AdamWOptimizer* opt) {
    for (auto& s : opt->states) {
        cudaFree(s.m);
        cudaFree(s.v);
    }
}
```

---

## `scheduler.h` / `scheduler.cpp`

### Concept

The learning rate schedule is a function of the current training step.
For transformers, the standard approach is:
- **Linear warmup:** ramp the LR from 0 to `max_lr` over `warmup_steps` steps.
  Without warmup, the large gradients at the start of training (random weights,
  large loss) can destabilize training.
- **Cosine decay:** after warmup, decay the LR following a cosine curve down
  to near-zero. This is smoother than step decay and tends to find better minima.

```
if step < warmup_steps:
    lr = max_lr * step / warmup_steps
else:
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * progress))
```

No GPU needed — this is pure scalar math on the CPU, called once per step.

### Header — `scheduler.h`

```cpp
#pragma once

struct LRScheduler {
    float max_lr;
    float min_lr;
    int   warmup_steps;
    int   total_steps;
};

LRScheduler scheduler_create(float max_lr, float min_lr,
                              int warmup_steps, int total_steps);
float scheduler_get_lr(const LRScheduler* s, int step);

// Call this every step to update the optimizer's learning rate
void scheduler_step(const LRScheduler* s, struct AdamWOptimizer* opt, int step);
```

### Implementation — `scheduler.cpp`

```cpp
#include "scheduler.h"
#include "../cuda/training/optimizer.cuh"
#include <math.h>

LRScheduler scheduler_create(float max_lr, float min_lr,
                              int warmup_steps, int total_steps)
{
    return { max_lr, min_lr, warmup_steps, total_steps };
}

float scheduler_get_lr(const LRScheduler* s, int step) {
    if (step < s->warmup_steps) {
        // Linear warmup
        return s->max_lr * (float)step / s->warmup_steps;
    } else {
        // Cosine decay
        float progress = (float)(step - s->warmup_steps)
                       / (float)(s->total_steps - s->warmup_steps);
        progress = fminf(1.0f, fmaxf(0.0f, progress));
        return s->min_lr + 0.5f * (s->max_lr - s->min_lr)
               * (1.0f + cosf(M_PI * progress));
    }
}

void scheduler_step(const LRScheduler* s, AdamWOptimizer* opt, int step) {
    opt->lr = scheduler_get_lr(s, step);
}
```

---

## `backward.cuh` / `backward.cu`

### Concept

Backpropagation is the chain rule applied recursively through the computation
graph. For each operation `y = f(x)`, backprop computes `dx = (dy/dx) * d_loss/dy`.

The full backward pass through one transformer block (in reverse order):

```
1. Residual (addition): gradient passes through unchanged — split into two paths
2. FFN Linear 2 backward: dW2 += ffn_mid^T * d_ffn_out, d_ffn_mid = d_ffn_out * W2^T
3. GELU backward: d_ffn_mid = d_ffn_mid * GELU'(pre_gelu_activations)
4. FFN Linear 1 backward: dW1 += ln2_out^T * d_pre_gelu, d_ln2_out = d_pre_gelu * W1^T
5. LayerNorm 2 backward
6. Residual 1 backward
7. Attention backward (see attention.cu)
8. LayerNorm 1 backward
```

The key operation repeated throughout is: given `C = A * B`, the backward pass gives:
```
dA = dC * B^T       (matmul with transposed B)
dB += A^T * dC      (accumulated outer product)
```

### Header — `backward.cuh`

```cpp
#pragma once
#include <cuda_runtime.h>
#include "transformer.cuh"

// Run the full backward pass through num_layers transformer blocks
// d_loss_output: gradient of loss w.r.t. transformer output (from decoder backward)
// Fills in all dW, db fields in params and writes d_encoder_output
void transformer_backward_full(
    float* d_encoder_output,          // gradient to pass back to encoder
    TransformerParams* params,
    const float* d_loss_output,
    const TransformerCache* cache,
    int batch, int seq_len
);

// Gradient clipping — prevents exploding gradients
// Scales all gradients so their global L2 norm <= max_norm
void clip_grad_norm(AdamWOptimizer* opt, float max_norm);
```

### Implementation — `backward.cu`

```cpp
#include "backward.cuh"
#include "../kernels/matmul.cuh"
#include "../kernels/layernorm.cuh"
#include "../kernels/activations.cuh"
#include <cuda_runtime.h>
#include <math.h>

// Compute L2 norm of a tensor (for gradient clipping)
__global__ void l2_norm_kernel(const float* x, float* norm_sq, int n) {
    extern __shared__ float sdata[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[threadIdx.x] = (i < n) ? x[i] * x[i] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(norm_sq, sdata[0]);
}

__global__ void scale_grads_kernel(float* x, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= scale;
}

void clip_grad_norm(AdamWOptimizer* opt, float max_norm) {
    // 1. Compute global norm across all parameter gradients
    float* d_norm_sq;
    cudaMalloc(&d_norm_sq, sizeof(float));
    cudaMemset(d_norm_sq, 0, sizeof(float));

    for (int i = 0; i < (int)opt->grads.size(); i++) {
        int n = opt->sizes[i];
        l2_norm_kernel<<<(n+255)/256, 256, 256*sizeof(float)>>>(
            opt->grads[i], d_norm_sq, n);
    }

    float norm_sq;
    cudaMemcpy(&norm_sq, d_norm_sq, sizeof(float), cudaMemcpyDeviceToHost);
    float norm = sqrtf(norm_sq);
    cudaFree(d_norm_sq);

    // 2. If norm exceeds max_norm, scale all gradients down
    if (norm > max_norm) {
        float scale = max_norm / norm;
        for (int i = 0; i < (int)opt->grads.size(); i++) {
            int n = opt->sizes[i];
            scale_grads_kernel<<<(n+255)/256, 256>>>(opt->grads[i], scale, n);
        }
    }
}

void transformer_backward_full(
    float* d_encoder_output, TransformerParams* params,
    const float* d_loss_output, const TransformerCache* cache,
    int batch, int seq_len)
{
    // Allocate a working gradient buffer (same size as one layer's activations)
    int N = batch * seq_len, D = params->d_model;
    float* d_current;
    cudaMalloc(&d_current, N * D * sizeof(float));
    cudaMemcpy(d_current, d_loss_output, N * D * sizeof(float), cudaMemcpyDeviceToDevice);

    for (int l = params->num_layers - 1; l >= 0; l--) {
        // Backprop through each sub-layer in reverse order
        // This is where you call layernorm_backward, attention_backward,
        // and the matmul gradient operations described above
        // Each step updates the dW/db fields in params->blocks[l]
        // and passes d_current backwards
    }

    cudaMemcpy(d_encoder_output, d_current, N * D * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(d_current);
}
```

---

# PART 4 — UTILS (`cuda/utils/`)

---

## `memory.cuh` / `memory.cu`

### Concept

Every `cudaMalloc` call is expensive — it synchronizes the GPU and may involve
a kernel to initialize memory. In a training loop where you allocate temporary
buffers for intermediate activations, doing this per-step kills performance.

An **arena allocator** solves this: allocate one large GPU buffer at startup
(e.g. 512MB), then sub-allocate from it by just bumping a pointer. Freeing is
O(1) — just reset the pointer to the arena start. Since training steps have
predictable memory layouts, you allocate the same set of tensors every step.

### Header — `memory.cuh`

```cpp
#pragma once
#include <cuda_runtime.h>
#include <cstddef>

struct GpuArena {
    float* base;         // start of GPU allocation
    size_t capacity;     // total bytes
    size_t used;         // bytes currently allocated
};

GpuArena arena_create(size_t bytes);
void     arena_destroy(GpuArena* a);
float*   arena_alloc(GpuArena* a, size_t n_floats);
void     arena_reset(GpuArena* a);   // free all sub-allocations at once
```

### Implementation — `memory.cu`

```cpp
#include "memory.cuh"
#include <stdio.h>
#include <stdlib.h>

GpuArena arena_create(size_t bytes) {
    GpuArena a;
    a.capacity = bytes;
    a.used = 0;
    cudaError_t err = cudaMalloc(&a.base, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "arena_create: cudaMalloc failed: %s\n",
                cudaGetErrorString(err));
        exit(1);
    }
    return a;
}

void arena_destroy(GpuArena* a) {
    cudaFree(a->base);
    a->base = nullptr;
    a->used = a->capacity = 0;
}

float* arena_alloc(GpuArena* a, size_t n_floats) {
    size_t bytes = n_floats * sizeof(float);
    // Align to 256 bytes (GPU memory access alignment)
    size_t aligned = (bytes + 255) & ~255UL;

    if (a->used + aligned > a->capacity) {
        fprintf(stderr, "arena_alloc: out of GPU memory (used=%zu cap=%zu req=%zu)\n",
                a->used, a->capacity, aligned);
        exit(1);
    }
    float* ptr = (float*)((char*)a->base + a->used);
    a->used += aligned;
    return ptr;
}

void arena_reset(GpuArena* a) {
    a->used = 0;   // all previous allocations are invalidated
}
```

---

## `checkpoint.h` / `checkpoint.cpp`

### Concept

Checkpointing saves all learned weight tensors to disk so you can:
- Resume training after interruption
- Load weights for inference
- Compare snapshots across training

The simplest format: a binary file with a small header (magic number, version,
num_layers, d_model, etc.) followed by raw float32 arrays for each weight tensor
in a fixed, deterministic order.

### Header — `checkpoint.h`

```cpp
#pragma once
#include <string>
#include "../cuda/model/transformer.cuh"
#include "../cuda/model/encoder.cuh"
#include "../cuda/model/decoder.cuh"

struct ModelConfig {
    int image_size, patch_size, channels;
    int d_model, num_heads, num_layers, d_ff;
    int batch_size;
};

// Save all parameters to a binary file
// Filename example: "checkpoints/epoch_005.bin"
void checkpoint_save(
    const std::string& path,
    const EncoderParams* enc,
    const TransformerParams* tr,
    const DecoderParams* dec,
    const ModelConfig& cfg,
    int step
);

// Load parameters from file into already-allocated GPU buffers
// Returns the step number from the checkpoint
int checkpoint_load(
    const std::string& path,
    EncoderParams* enc,
    TransformerParams* tr,
    DecoderParams* dec,
    ModelConfig* cfg
);

bool checkpoint_exists(const std::string& path);
```

### Implementation — `checkpoint.cpp`

```cpp
#include "checkpoint.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define MAGIC   0x54524E47u  // "TRNG"
#define VERSION 1

// Helper: copy GPU tensor to CPU buffer and write to file
static void write_tensor(FILE* f, const float* d_ptr, int n) {
    std::vector<float> buf(n);
    cudaMemcpy(buf.data(), d_ptr, n * sizeof(float), cudaMemcpyDeviceToHost);
    fwrite(buf.data(), sizeof(float), n, f);
}

// Helper: read from file into GPU tensor
static void read_tensor(FILE* f, float* d_ptr, int n) {
    std::vector<float> buf(n);
    fread(buf.data(), sizeof(float), n, f);
    cudaMemcpy(d_ptr, buf.data(), n * sizeof(float), cudaMemcpyHostToDevice);
}

void checkpoint_save(
    const std::string& path,
    const EncoderParams* enc, const TransformerParams* tr,
    const DecoderParams* dec, const ModelConfig& cfg, int step)
{
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return; }

    // Header
    fwrite(&MAGIC,   sizeof(uint32_t), 1, f);
    fwrite(&VERSION, sizeof(int),      1, f);
    fwrite(&cfg,     sizeof(ModelConfig), 1, f);
    fwrite(&step,    sizeof(int),      1, f);

    int D  = cfg.d_model, PD = cfg.patch_size * cfg.patch_size * cfg.channels;
    int NP = (cfg.image_size / cfg.patch_size) * (cfg.image_size / cfg.patch_size);

    // Encoder weights
    write_tensor(f, enc->patch_proj_W, PD * D);
    write_tensor(f, enc->patch_proj_b, D);
    write_tensor(f, enc->pos_embedding, NP * D);

    // Transformer weights (all layers)
    for (int l = 0; l < tr->num_layers; l++) {
        const TransformerBlockParams& bp = tr->blocks[l];
        write_tensor(f, bp.attn.W_Q, D * D);
        write_tensor(f, bp.attn.W_K, D * D);
        write_tensor(f, bp.attn.W_V, D * D);
        write_tensor(f, bp.attn.W_O, D * D);
        // biases, layer norm params, FFN weights...
    }

    // Decoder weights
    write_tensor(f, dec->proj_W, D * PD);
    write_tensor(f, dec->proj_b, PD);

    fclose(f);
    printf("Saved checkpoint: %s (step %d)\n", path.c_str(), step);
}

int checkpoint_load(
    const std::string& path,
    EncoderParams* enc, TransformerParams* tr,
    DecoderParams* dec, ModelConfig* cfg)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return -1; }

    uint32_t magic; int version;
    fread(&magic,   sizeof(uint32_t),   1, f);
    fread(&version, sizeof(int),        1, f);
    fread(cfg,      sizeof(ModelConfig),1, f);
    int step; fread(&step, sizeof(int), 1, f);

    if (magic != MAGIC) { fprintf(stderr, "bad checkpoint magic\n"); fclose(f); return -1; }

    int D = cfg->d_model, PD = cfg->patch_size * cfg->patch_size * cfg->channels;
    int NP = (cfg->image_size / cfg->patch_size) * (cfg->image_size / cfg->patch_size);

    read_tensor(f, enc->patch_proj_W, PD * D);
    read_tensor(f, enc->patch_proj_b, D);
    read_tensor(f, enc->pos_embedding, NP * D);

    for (int l = 0; l < tr->num_layers; l++) {
        TransformerBlockParams& bp = tr->blocks[l];
        read_tensor(f, bp.attn.W_Q, D * D);
        read_tensor(f, bp.attn.W_K, D * D);
        read_tensor(f, bp.attn.W_V, D * D);
        read_tensor(f, bp.attn.W_O, D * D);
    }

    read_tensor(f, dec->proj_W, D * PD);
    read_tensor(f, dec->proj_b, PD);

    fclose(f);
    printf("Loaded checkpoint: %s (step %d)\n", path.c_str(), step);
    return step;
}

bool checkpoint_exists(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (f) { fclose(f); return true; }
    return false;
}
```

---

# PART 5 — C++ LAYER (`cpp/`)

---

## `dataloader.h` / `dataloader.cpp`

### Concept

The DataLoader's job is to feed the GPU with batches of images without
letting it sit idle. It runs a background thread that prefetches the next
batch while the current batch is training — the GPU and CPU/disk overlap.

The loading pipeline:
1. **Shuffle** an index array at the start of each epoch
2. **Background thread** reads files, applies augmentations, copies to a
   pinned CPU buffer (pinned = page-locked RAM, enables async GPU transfers)
3. **Main thread** calls `next_batch()`, which initiates a `cudaMemcpyAsync`
   from the pinned buffer to a GPU buffer, returning a device pointer

**Augmentations** (applied on CPU):
- Random horizontal flip: swap left/right halves of the image array
- Random crop: pick a random sub-region, then resize back to target
- Normalize: `(pixel / 255.0f - mean) / std`

### Header — `dataloader.h`

```cpp
#pragma once
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cuda_runtime.h>

struct AugConfig {
    bool   random_flip   = true;
    float  crop_scale    = 0.9f;    // crop to 90% then resize
    float  mean[3]       = {0.5f, 0.5f, 0.5f};
    float  std[3]        = {0.5f, 0.5f, 0.5f};
};

class DataLoader {
public:
    DataLoader(const std::string& data_dir,
               int batch_size, int image_size, int channels,
               const AugConfig& aug = {});
    ~DataLoader();

    // Returns a device pointer to the next batch: (batch x C x H x W)
    // Blocks until the batch is ready
    float* next_batch();

    int num_batches() const;
    void reset();    // reshuffle and restart epoch

private:
    void prefetch_worker();
    void load_and_augment(const std::string& path, float* out);
    void augment(float* img);

    std::string  data_dir_;
    int          batch_size_, image_size_, channels_;
    AugConfig    aug_;

    std::vector<std::string> file_list_;
    std::vector<int>         indices_;
    int                      current_idx_;

    // Double-buffering: one buffer being consumed, one being filled
    float* pinned_buf_[2];   // page-locked CPU buffers
    float* gpu_buf_[2];      // corresponding GPU buffers
    int    active_buf_;

    cudaStream_t transfer_stream_;

    std::thread             worker_;
    std::mutex              mutex_;
    std::condition_variable cv_ready_, cv_consumed_;
    bool                    next_ready_ = false;
    bool                    stop_       = false;
};
```

### Implementation — `dataloader.cpp`

```cpp
#include "dataloader.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../vendor/stb_image.h"
#include <filesystem>
#include <algorithm>
#include <random>
#include <cstring>

namespace fs = std::filesystem;

DataLoader::DataLoader(const std::string& dir,
                       int batch_size, int image_size, int channels,
                       const AugConfig& aug)
    : data_dir_(dir), batch_size_(batch_size),
      image_size_(image_size), channels_(channels), aug_(aug),
      current_idx_(0), active_buf_(0)
{
    // Gather all image file paths
    for (auto& entry : fs::directory_iterator(dir))
        if (entry.path().extension() == ".jpg" ||
            entry.path().extension() == ".png")
            file_list_.push_back(entry.path().string());

    indices_.resize(file_list_.size());
    std::iota(indices_.begin(), indices_.end(), 0);

    size_t batch_bytes = (size_t)batch_size * channels * image_size * image_size
                         * sizeof(float);

    // Allocate pinned (page-locked) CPU buffers — faster DMA to GPU
    cudaMallocHost(&pinned_buf_[0], batch_bytes);
    cudaMallocHost(&pinned_buf_[1], batch_bytes);
    cudaMalloc(&gpu_buf_[0], batch_bytes);
    cudaMalloc(&gpu_buf_[1], batch_bytes);

    cudaStreamCreate(&transfer_stream_);

    // Start prefetch thread
    worker_ = std::thread(&DataLoader::prefetch_worker, this);
}

DataLoader::~DataLoader() {
    { std::lock_guard<std::mutex> lk(mutex_); stop_ = true; }
    cv_consumed_.notify_all();
    worker_.join();
    cudaFreeHost(pinned_buf_[0]); cudaFreeHost(pinned_buf_[1]);
    cudaFree(gpu_buf_[0]);        cudaFree(gpu_buf_[1]);
    cudaStreamDestroy(transfer_stream_);
}

void DataLoader::load_and_augment(const std::string& path, float* out) {
    int w, h, c;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &c, channels_);
    if (!data) return;

    // Resize to image_size_ x image_size_ using nearest-neighbor (simple)
    // In production: use bilinear interpolation
    for (int ch = 0; ch < channels_; ch++)
        for (int y = 0; y < image_size_; y++)
            for (int x = 0; x < image_size_; x++) {
                int src_y = y * h / image_size_;
                int src_x = x * w / image_size_;
                int src_idx = (src_y * w + src_x) * channels_ + ch;
                // Normalize to [-1, 1]
                out[ch * image_size_ * image_size_ + y * image_size_ + x]
                    = (data[src_idx] / 255.0f - aug_.mean[ch]) / aug_.std[ch];
            }

    stbi_image_free(data);
    augment(out);   // apply random augmentations
}

void DataLoader::augment(float* img) {
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // Random horizontal flip
    if (aug_.random_flip && dist(rng) > 0.5f) {
        int H = image_size_, W = image_size_;
        for (int c = 0; c < channels_; c++)
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W / 2; x++) {
                    float tmp = img[c*H*W + y*W + x];
                    img[c*H*W + y*W + x]       = img[c*H*W + y*W + (W-1-x)];
                    img[c*H*W + y*W + (W-1-x)] = tmp;
                }
    }
}

void DataLoader::prefetch_worker() {
    int fill_buf = 1;   // fill the non-active buffer
    while (true) {
        {
            std::unique_lock<std::mutex> lk(mutex_);
            cv_consumed_.wait(lk, [&]{ return !next_ready_ || stop_; });
            if (stop_) return;
        }

        // Load a batch into the fill buffer
        int bsz = batch_size_;
        size_t img_floats = (size_t)channels_ * image_size_ * image_size_;
        for (int i = 0; i < bsz; i++) {
            if (current_idx_ >= (int)indices_.size()) {
                reset();
            }
            std::string path = file_list_[indices_[current_idx_++]];
            load_and_augment(path, pinned_buf_[fill_buf] + i * img_floats);
        }

        // Async copy pinned CPU → GPU
        size_t bytes = (size_t)bsz * img_floats * sizeof(float);
        cudaMemcpyAsync(gpu_buf_[fill_buf], pinned_buf_[fill_buf],
                        bytes, cudaMemcpyHostToDevice, transfer_stream_);
        cudaStreamSynchronize(transfer_stream_);

        { std::lock_guard<std::mutex> lk(mutex_); next_ready_ = true; fill_buf ^= 1; }
        cv_ready_.notify_one();
    }
}

float* DataLoader::next_batch() {
    std::unique_lock<std::mutex> lk(mutex_);
    cv_ready_.wait(lk, [&]{ return next_ready_; });
    next_ready_ = false;
    active_buf_ ^= 1;
    cv_consumed_.notify_one();
    return gpu_buf_[active_buf_];
}

void DataLoader::reset() {
    std::shuffle(indices_.begin(), indices_.end(),
                 std::mt19937{std::random_device{}()});
    current_idx_ = 0;
}

int DataLoader::num_batches() const {
    return (int)file_list_.size() / batch_size_;
}
```

---

## `preprocess.h` / `preprocess.cpp`

### Concept

A one-time pass over your raw image dataset that:
1. Decodes every JPEG/PNG
2. Resizes to the training resolution
3. Saves as raw float32 binary arrays

Doing this upfront means the DataLoader reads simple flat binary files instead
of decoding JPEGs during training — much faster I/O, especially for small images.

### Header — `preprocess.h`

```cpp
#pragma once
#include <string>

struct PreprocessConfig {
    int target_size  = 64;     // resize images to target_size x target_size
    int channels     = 3;
    bool normalize   = true;   // map pixels to [-1, 1]
};

// Process all images in src_dir, write .bin files to dst_dir
void preprocess_dataset(
    const std::string& src_dir,
    const std::string& dst_dir,
    const PreprocessConfig& cfg
);
```

### Implementation — `preprocess.cpp`

```cpp
#include "preprocess.h"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../vendor/stb_image.h"
#include "../vendor/stb_image_resize.h"   // stb_image_resize2.h in newer versions
#include <filesystem>
#include <fstream>
#include <stdio.h>

namespace fs = std::filesystem;

void preprocess_dataset(
    const std::string& src_dir,
    const std::string& dst_dir,
    const PreprocessConfig& cfg)
{
    fs::create_directories(dst_dir);
    int S = cfg.target_size, C = cfg.channels;

    std::vector<unsigned char> resized(S * S * C);
    std::vector<float> normalized(S * S * C);

    int count = 0;
    for (auto& entry : fs::directory_iterator(src_dir)) {
        std::string ext = entry.path().extension().string();
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png") continue;

        int w, h, c;
        unsigned char* data = stbi_load(entry.path().c_str(), &w, &h, &c, C);
        if (!data) { fprintf(stderr, "skip %s\n", entry.path().c_str()); continue; }

        // Resize using stb_image_resize
        stbir_resize_uint8(data, w, h, 0, resized.data(), S, S, 0, C);
        stbi_image_free(data);

        // Convert to float, channels-first (C x H x W), normalize to [-1, 1]
        for (int ch = 0; ch < C; ch++)
            for (int y = 0; y < S; y++)
                for (int x = 0; x < S; x++) {
                    unsigned char px = resized[(y * S + x) * C + ch];
                    float val = cfg.normalize ? (px / 127.5f - 1.0f) : (px / 255.0f);
                    normalized[ch * S * S + y * S + x] = val;
                }

        // Write to .bin file — just raw floats, no header
        std::string out_path = dst_dir + "/" + entry.path().stem().string() + ".bin";
        std::ofstream out(out_path, std::ios::binary);
        out.write((char*)normalized.data(), normalized.size() * sizeof(float));
        count++;

        if (count % 100 == 0)
            printf("Processed %d images...\n", count);
    }
    printf("Done. Processed %d images -> %s\n", count, dst_dir.c_str());
}
```

---

## `logger.h` / `logger.cpp`

### Concept

During training you need to track:
- **Loss curve** — written to a CSV so you can plot it
- **Sample images** — saved every N steps so you can visually check if the
  model is learning to reconstruct images

Sample images are the most important debug tool — if reconstruction looks
like noise after 1000 steps on a simple dataset, something is wrong
(exploding gradients, wrong loss, bug in the encoder/decoder).

### Header — `logger.h`

```cpp
#pragma once
#include <string>
#include <fstream>
#include <cuda_runtime.h>

class Logger {
public:
    Logger(const std::string& log_dir, int save_every_n_steps = 500);
    ~Logger();

    void log_loss(int step, int epoch, float loss, float lr);

    // Save a batch of reconstructed images to PNG
    // pred: (batch x C x H x W) device pointer, values in [-1, 1]
    void save_samples(int step, const float* d_pred, const float* d_target,
                      int batch, int channels, int image_size);

private:
    std::string  log_dir_;
    int          save_every_;
    std::ofstream loss_csv_;
};
```

### Implementation — `logger.cpp`

```cpp
#include "logger.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../vendor/stb_image_write.h"
#include <filesystem>
#include <vector>
#include <stdio.h>

namespace fs = std::filesystem;

Logger::Logger(const std::string& log_dir, int save_every)
    : log_dir_(log_dir), save_every_(save_every)
{
    fs::create_directories(log_dir);
    fs::create_directories(log_dir + "/samples");
    loss_csv_.open(log_dir + "/loss.csv");
    loss_csv_ << "step,epoch,loss,lr\n";
}

Logger::~Logger() { loss_csv_.close(); }

void Logger::log_loss(int step, int epoch, float loss, float lr) {
    loss_csv_ << step << "," << epoch << "," << loss << "," << lr << "\n";
    loss_csv_.flush();
    printf("[step %6d | epoch %3d] loss=%.6f lr=%.2e\n", step, epoch, loss, lr);
}

void Logger::save_samples(
    int step, const float* d_pred, const float* d_target,
    int batch, int C, int H)
{
    if (step % save_every_ != 0) return;

    int n = batch * C * H * H;
    std::vector<float> h_pred(n), h_target(n);
    cudaMemcpy(h_pred.data(),   d_pred,   n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_target.data(), d_target, n*sizeof(float), cudaMemcpyDeviceToHost);

    // Save first 4 images from the batch (predicted + target side by side)
    int save_n = std::min(batch, 4);
    for (int b = 0; b < save_n; b++) {
        // Convert CHW float [-1,1] -> HWC uint8 [0,255]
        std::vector<unsigned char> pred_u8(H * H * C);
        std::vector<unsigned char> tgt_u8(H * H * C);
        for (int ch = 0; ch < C; ch++)
            for (int y = 0; y < H; y++)
                for (int x = 0; x < H; x++) {
                    int ci = ch*H*H + y*H + x;
                    int hw = (y*H + x)*C + ch;
                    pred_u8[hw] = (unsigned char)((h_pred  [b*C*H*H + ci] + 1.0f) * 127.5f);
                    tgt_u8 [hw] = (unsigned char)((h_target[b*C*H*H + ci] + 1.0f) * 127.5f);
                }

        char path[256];
        snprintf(path, 256, "%s/samples/step%06d_b%d_pred.png",
                 log_dir_.c_str(), step, b);
        stbi_write_png(path, H, H, C, pred_u8.data(), H * C);

        snprintf(path, 256, "%s/samples/step%06d_b%d_target.png",
                 log_dir_.c_str(), step, b);
        stbi_write_png(path, H, H, C, tgt_u8.data(), H * C);
    }
    printf("Saved sample images at step %d\n", step);
}
```

---

## `train.cpp`

### Concept

The main training loop. Ties every module together:

```
for each epoch:
    for each batch:
        1. Get batch from DataLoader          (GPU pointer)
        2. Encoder forward                    (images -> tokens)
        3. Transformer forward                (tokens -> tokens)
        4. Decoder forward                    (tokens -> reconstructed image)
        5. Compute loss + gradient
        6. Decoder backward
        7. Transformer backward
        8. Encoder backward
        9. Clip gradients
       10. Optimizer step
       11. Zero gradients
       12. Update LR scheduler
       13. Log loss, save samples, save checkpoint
```

```cpp
#include <stdio.h>
#include <cuda_runtime.h>
#include "../cuda/model/encoder.cuh"
#include "../cuda/model/transformer.cuh"
#include "../cuda/model/decoder.cuh"
#include "../cuda/training/optimizer.cuh"
#include "../cuda/training/loss.cuh"
#include "../cuda/training/scheduler.h"
#include "../cuda/training/backward.cuh"
#include "../cuda/utils/checkpoint.h"
#include "../cuda/utils/memory.cuh"
#include "dataloader.h"
#include "logger.h"
#define NLOHMANN_JSON_HPP
#include "../vendor/nlohmann/json.hpp"
#include <fstream>

int main(int argc, char** argv) {
    // ---- Load config ----
    std::ifstream f("config.json");
    nlohmann::json cfg_json = nlohmann::json::parse(f);

    ModelConfig cfg;
    cfg.image_size  = cfg_json["image_size"];
    cfg.patch_size  = cfg_json["patch_size"];
    cfg.channels    = cfg_json["channels"];
    cfg.d_model     = cfg_json["d_model"];
    cfg.num_heads   = cfg_json["num_heads"];
    cfg.num_layers  = cfg_json["num_layers"];
    cfg.d_ff        = cfg_json["d_ff"];
    cfg.batch_size  = cfg_json["batch_size"];

    int total_steps    = cfg_json["total_steps"];
    int warmup_steps   = cfg_json["warmup_steps"];
    float max_lr       = cfg_json["max_lr"];
    float min_lr       = cfg_json["min_lr"];
    float weight_decay = cfg_json["weight_decay"];
    int   save_every   = cfg_json["save_every"];

    // ---- Allocate GPU memory for all parameters ----
    // (allocate each weight tensor via cudaMalloc, initialize with Kaiming/Xavier)
    EncoderParams     enc_params   = {};  // fill in allocations
    TransformerParams tr_params    = {};
    DecoderParams     dec_params   = {};

    // encoder_init(&enc_params, ...);
    // transformer_init(&tr_params, cfg);
    // decoder_init(&dec_params, cfg);

    // ---- Set up optimizer ----
    AdamWOptimizer opt = adamw_create(max_lr, 0.9f, 0.999f, 1e-8f, weight_decay);
    // Register every weight tensor:
    // adamw_add_param(&opt, enc_params.patch_proj_W, enc_params.d_patch_proj_W, PD*D);
    // ... etc for every parameter

    LRScheduler scheduler = scheduler_create(max_lr, min_lr, warmup_steps, total_steps);

    // ---- Data ----
    DataLoader loader("data/processed", cfg.batch_size,
                      cfg.image_size, cfg.channels);
    Logger logger("logs", save_every);

    // ---- Activation buffers (GPU) ----
    int N  = cfg.batch_size * (cfg.image_size/cfg.patch_size) * (cfg.image_size/cfg.patch_size);
    int D  = cfg.d_model, C = cfg.channels, H = cfg.image_size;
    float *d_tokens, *d_tr_out, *d_recon, *d_loss, *d_loss_grad;
    cudaMalloc(&d_tokens,    N * D * sizeof(float));
    cudaMalloc(&d_tr_out,    N * D * sizeof(float));
    cudaMalloc(&d_recon,     cfg.batch_size * C * H * H * sizeof(float));
    cudaMalloc(&d_loss,      sizeof(float));
    cudaMalloc(&d_loss_grad, cfg.batch_size * C * H * H * sizeof(float));

    TransformerCache tr_cache = {};   // allocate all intermediate buffers

    // ---- Resume from checkpoint if available ----
    int start_step = 0;
    if (checkpoint_exists("checkpoints/latest.bin"))
        start_step = checkpoint_load("checkpoints/latest.bin",
                                     &enc_params, &tr_params, &dec_params, &cfg);

    // ---- Training loop ----
    int step = start_step;
    for (int epoch = 0; step < total_steps; epoch++) {
        for (int b = 0; b < loader.num_batches() && step < total_steps; b++, step++) {

            float* d_images = loader.next_batch();   // (batch x C x H x W) on GPU

            // Forward pass
            encoder_forward(d_tokens, d_images, &enc_params, /* cfg */ nullptr, cfg.batch_size);
            transformer_forward(d_tr_out, &tr_cache, d_tokens, &tr_params, cfg.batch_size, N/cfg.batch_size);
            decoder_forward(d_recon, d_tr_out, &dec_params, /* cfg */ nullptr, cfg.batch_size);

            // Loss
            mse_loss(d_loss, d_loss_grad, d_recon, d_images, cfg.batch_size * C * H * H);

            // Read loss to CPU for logging
            float loss_val;
            cudaMemcpy(&loss_val, d_loss, sizeof(float), cudaMemcpyDeviceToHost);

            // Backward pass
            // decoder_backward(...)
            // transformer_backward_full(...)
            // encoder_backward(...)

            // Gradient clipping and optimizer step
            clip_grad_norm(&opt, 1.0f);
            adamw_step(&opt);
            adamw_zero_grad(&opt);

            // LR update
            scheduler_step(&scheduler, &opt, step);

            // Logging
            logger.log_loss(step, epoch, loss_val, opt.lr);
            logger.save_samples(step, d_recon, d_images, cfg.batch_size, C, H);

            // Checkpoint
            if (step % save_every == 0) {
                char ckpt_path[256];
                snprintf(ckpt_path, 256, "checkpoints/step_%06d.bin", step);
                checkpoint_save(ckpt_path, &enc_params, &tr_params, &dec_params, cfg, step);
                checkpoint_save("checkpoints/latest.bin", &enc_params, &tr_params, &dec_params, cfg, step);
            }
        }
    }

    printf("Training complete.\n");
    // cleanup: cudaFree everything, adamw_destroy(&opt)
    return 0;
}
```

---

## `generate.cpp`

### Concept

Inference mode: load a checkpoint, then generate or reconstruct images.
For a pure autoencoder, you pass a real image through and observe the
reconstruction. For a generative model (VAE / masked autoencoder), you
pass in a noisy or masked version and observe what the model fills in.

```cpp
#include <stdio.h>
#include "../cuda/utils/checkpoint.h"
#include "../cuda/model/encoder.cuh"
#include "../cuda/model/transformer.cuh"
#include "../cuda/model/decoder.cuh"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../vendor/stb_image.h"
#include "../vendor/stb_image_write.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: generate <checkpoint.bin> <input_image.jpg> [output.png]\n");
        return 1;
    }

    ModelConfig cfg;
    EncoderParams  enc = {};
    TransformerParams tr = {};
    DecoderParams  dec = {};

    int step = checkpoint_load(argv[1], &enc, &tr, &dec, &cfg);
    printf("Loaded checkpoint from step %d\n", step);

    // Load and preprocess input image
    int w, h, c;
    unsigned char* img = stbi_load(argv[2], &w, &h, &c, cfg.channels);
    // resize, normalize, copy to GPU...

    float *d_input, *d_tokens, *d_tr_out, *d_output;
    int img_size = cfg.batch_size * cfg.channels * cfg.image_size * cfg.image_size;
    cudaMalloc(&d_input,  img_size * sizeof(float));
    cudaMalloc(&d_tokens, /* ... */ 0);
    cudaMalloc(&d_tr_out, /* ... */ 0);
    cudaMalloc(&d_output, img_size * sizeof(float));

    // Forward pass (no backward needed)
    encoder_forward(d_tokens, d_input, &enc, nullptr, 1);
    transformer_forward(d_tr_out, nullptr, d_tokens, &tr, 1, cfg.num_patches);
    decoder_forward(d_output, d_tr_out, &dec, nullptr, 1);

    // Copy output to CPU and save
    std::vector<float> h_out(img_size);
    cudaMemcpy(h_out.data(), d_output, img_size*sizeof(float), cudaMemcpyDeviceToHost);

    // Convert [-1,1] float CHW -> [0,255] uint8 HWC
    int H = cfg.image_size, C = cfg.channels;
    std::vector<unsigned char> out_img(H * H * C);
    for (int ch = 0; ch < C; ch++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < H; x++) {
                float v = h_out[ch*H*H + y*H + x];
                out_img[(y*H+x)*C+ch] = (unsigned char)((v+1.0f)*127.5f);
            }

    const char* out_path = (argc > 3) ? argv[3] : "output.png";
    stbi_write_png(out_path, H, H, C, out_img.data(), H * C);
    printf("Saved generated image: %s\n", out_path);

    stbi_image_free(img);
    return 0;
}
```

---

# PART 6 — BUILD SYSTEM

## `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.20)
project(transformer_imggen LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Set your GPU architecture: sm_86 = RTX 3000 series, sm_89 = RTX 4000 series
# Check yours with: nvidia-smi --query-gpu=compute_cap --format=csv
set(CMAKE_CUDA_ARCHITECTURES 86)

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -lineinfo")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -O3 -Wall")

# Vendor include path
include_directories(vendor)

# ---- CUDA kernel library ----
add_library(cuda_kernels STATIC
    cuda/kernels/matmul.cu
    cuda/kernels/softmax.cu
    cuda/kernels/layernorm.cu
    cuda/kernels/activations.cu
    cuda/kernels/attention.cu
)
set_target_properties(cuda_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ---- CUDA model library ----
add_library(cuda_model STATIC
    cuda/model/encoder.cu
    cuda/model/decoder.cu
    cuda/model/transformer.cu
)
target_link_libraries(cuda_model cuda_kernels)
set_target_properties(cuda_model PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ---- CUDA training library ----
add_library(cuda_training STATIC
    cuda/training/optimizer.cu
    cuda/training/loss.cu
    cuda/training/backward.cu
    cuda/training/scheduler.cpp
    cuda/utils/memory.cu
    cuda/utils/checkpoint.cpp
)
target_link_libraries(cuda_training cuda_model)

# ---- Trainer executable ----
add_executable(train
    cpp/train.cpp
    cpp/dataloader.cpp
    cpp/logger.cpp
)
target_link_libraries(train cuda_training cuda_model cuda_kernels)

# ---- Preprocessor executable ----
add_executable(preprocess
    cpp/preprocess.cpp
)

# ---- Generator executable ----
add_executable(generate
    cpp/generate.cpp
)
target_link_libraries(generate cuda_training cuda_model cuda_kernels)
```

## `config.json`

```json
{
    "image_size":    64,
    "patch_size":     8,
    "channels":       3,
    "d_model":      256,
    "num_heads":      8,
    "num_layers":     6,
    "d_ff":        1024,
    "batch_size":    32,
    "total_steps": 50000,
    "warmup_steps":  500,
    "max_lr":       3e-4,
    "min_lr":       1e-5,
    "weight_decay": 0.01,
    "save_every":  1000
}
```

---

# Summary: Build and run order

```bash
# 1. Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

 cmake -B build 
 cmake --build build --config Release


# 2. Preprocess your images
.\Release\preprocess.exe ..\data\raw ..\data\processed  # reads data/raw/, writes data/processed/

# 3. Train
.\Release\train.exe 
Remove-Item checkpoints\* -Force; .\build\Release\train.exe
      # reads config.json, logs to logs/, saves to checkpoints/

# 4. Generate / reconstruct
./generate checkpoints/latest.bin path/to/image.jpg output.png
```

The expected learning curve: loss should drop significantly in the first
500–1000 steps, then slowly improve. If loss is NaN after step 1, check
for missing gradient zero-ing, too-high learning rate, or a bug in the
backward pass (verify each gradient numerically against finite differences
on a toy input before full training).

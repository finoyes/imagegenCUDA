// softmax function -> converts the raw attention scores to probabilities.

#include "softmax.cuh"
#include <cuda_runtime.h>

__global__ void softmax_kernel(float*x, int cols){
    extern __shared__ float sdata[]; // declares a shared meemory array whose size is NOT known at compile time.
    int row = blockIdx.x;
    float* row_ptr = x + row * cols; // pointer to the start of the current block's row.

    float thread_max = -1e38f;// any real val will be larger than this
    for(int i = threadIdx.x; i < cols; i += blockDim.x){
        thread_max = fmaxf(thread_max, row_ptr[i]);
    }
    sdata[threadIdx.x] = thread_max;// storing or writing each thread's pertial memory in the shared memory.
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){

        if(threadIdx.x < stride){
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float row_max = sdata[0]; 
    /*parallel tree reduction - > finds maximum across blocks 
    iteration1: stride = 128. thrds 0..127 compare with thread 128 positions ahead. after this sdata[0..127]each hold max of 2 vals.
    iteration2: stride = 64. thrds 0..63 compare with thread 64 positions ahead. after this sdata[0..63]each hold max of 4 vals.
    continue till stride= 1. after log2(256)=8 iterations, sdata[0] holds all 256 max values.
    stride >>= 1 = /2. __syncthreads() in the loop required becaude each iteration reads vals written by the previous iteration. without?, race conditons.
    after loop, sdata[0] is the row max, all thread read is as row_max.
    */

    float thrd_sum = 0.0f;
    for(int i = threadIdx.x; i < cols; i += blockDim.x){
        float val = expf(row_ptr[i] - row_max);
        row_ptr[i] = val;
        thrd_sum += val;
    }// exp(x-x_max) (overflow avoidence), override input, and accumulate the sum.
    
    sdata[threadIdx.x] = thrd_sum;
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if(threadIdx.x < stride){
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }// another parallel reduction same as above, now for sum instead of max
    float row_sum = sdata[0];

    for(int i = threadIdx.x; i < cols; i += blockDim.x){
        row_ptr[i] /= row_sum;
    }// normalize the exp values by the sum to get the final softmax output. the results are probability distributions: all values in [0,1], summing to 1.0 per row.
}

void softmax(float* input, int rows, int cols){

    int threads = 256;
    size_t shared = threads * sizeof(float); // shared = 256 x 4 = 1024bytes of shared mem —for our sdata array.
    softmax_kernel<<<rows, threads, shared>>>(input, cols); // The third kernel argument shared is what extern __shared__ float sdata[] actually gets. The kernel sees it as an array of 256 floats.

}

void attention_softmax(float* scores, int batch, int heads, int seq_len){
    int rows = batch * heads * seq_len;
    softmax(scores, rows, seq_len);
}

// Backward through row-wise softmax.
// s:   post-softmax weights (rows × cols), already computed in forward pass.
// ds:  upstream gradient (rows × cols).
// dx:  output gradient (rows × cols), overwritten.
// Formula per row: dx[i] = s[i] * (ds[i] - sum_j(ds[j] * s[j]))
__global__ void softmax_backward_kernel(float* dx, const float* s, const float* ds, int cols) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    const float* s_row  = s  + row * cols;
    const float* ds_row = ds + row * cols;
    float*       dx_row = dx + row * cols;

    // Compute dot product sum_j(ds[j] * s[j]) for this row
    float dot = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        dot += ds_row[i] * s_row[i];
    sdata[threadIdx.x] = dot;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();
    }
    float row_dot = sdata[0];

    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        dx_row[i] = s_row[i] * (ds_row[i] - row_dot);
}

void attention_softmax_backward(float* dx, const float* s, const float* ds,
                                int batch, int heads, int seq_len)
{
    int rows    = batch * heads * seq_len;
    int threads = 256;
    size_t shared = threads * sizeof(float);
    softmax_backward_kernel<<<rows, threads, shared>>>(dx, s, ds, seq_len);
}

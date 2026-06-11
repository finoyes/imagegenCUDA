#include "attention.cuh"
#include "matmul.cuh"
#include "softmax.cuh"
#include <cuda_runtime.h>
#include <math.h>

__global__ void bias_summ_krln(float* output, const float* bias, int rows, int cols){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * cols){
        output[i] += bias[i % cols];// maps any flat index back to a col index(thrd 0 col 0, thrd 512  col 0 row 2)
    }

}// broadcasts (cols,) bias vec acrs all rows. += — adds bias to existing values (from the matmul result).

__global__ void scale_kernel(float* x, float scale, int n){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        x[i] *= scale;
    }
}// * every ele by scale, used to apply the 1/sqrt(d_k) factor to attention scores.

__global__ void transpose_kernel(float* output, const float* input, int batch, int heads, int seq, int dk ){

    int b = blockIdx.z;
    int h = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < seq * dk){
        int s = i / dk /* seq position in output*/, k = i % dk;//depth dimension in the output
        output[b*heads*seq*dk + h*seq*dk + s*dk + k] = input[b*heads*dk*seq + h*dk*seq + k*seq + s];
    }

}// transposes the last two dims ofaa 4D tensor ( b,h,s,d ) -> (b,h,d,s).

void attention_forward(float* output, AttentionCache* cache, const float* input, const AttentionParams* p, 
                       int batch, int seq_len, int d_model, int num_heads){

                        int d_k    = d_model / num_heads;// d_k -> dimentsional subspace, each head operated on it.
                        int total  = batch * seq_len;// flatten batch and seq into 1D, matrixmult treats this as a 2D where rows are all tokens from all batch items,
                        int threads = 256;  
                        
                        matrixmult(input, p->W_Q, cache->Q, total, d_model, d_model);
                        matrixmult(input, p->W_K, cache->K, total, d_model, d_model);
                        matrixmult(input, p->W_V, cache->V, total, d_model, d_model);
//Three separate matrix multiplications, each (total × d_model) × (d_model × d_model) = (total × d_model). result stored in cache for backward pass use. most expensive in attention.

                        int n_qkv = total * d_model;// total elements to process.
                        bias_summ_krln<<<(n_qkv+threads-1)/threads, threads>>>(cache->Q, p->b_Q, total, d_model); 
                        bias_summ_krln<<<(n_qkv+threads-1)/threads, threads>>>(cache->K, p->b_K, total, d_model);
                        bias_summ_krln<<<(n_qkv+threads-1)/threads, threads>>>(cache->V, p->b_V, total, d_model);
                        //adds bias to each projection, launches one per projection.

                        float scale = 1.0f / sqrtf((float)d_k); // cast to float before sqrtf to avoid int sqrt, this scale factor prevemts the dot products from growing too large as d_k increases, which would push softmax into saturation and kill gradients.

                        batched_matrixmult(cache->Q, cache->K, cache->scores, batch * num_heads, seq_len, d_k, seq_len);
/*compute Q*K^T for every batch itemand every head simultaneously, Treating Q as (batch*heads × seq × d_k) and K as (batch*heads × d_k × seq).
Output scores is (batch*heads × seq × seq) — the raw attention scores.
*/

                        int n_scores = batch * num_heads * seq_len * seq_len;
                        scale_kernel<<<(n_scores+threads-1)/threads, threads>>>(cache->scores, scale, n_scores); // scales all b * h * seq * seq score val by 1/sqrt(d_k).


                        attention_softmax(cache->scores, batch, num_heads, seq_len);/*applies softmax along the last dimension( the "key" dimension) of each score row. after this each row of seqxseq mtx sums to 1 - attention wieghts.*/

                        batched_matrixmult(cache->scores, cache->V, output, batch * num_heads, seq_len, seq_len, d_k);/*compute weighted sum of values: attention score * V, (b, h, s, s) x (b, h, s, d_k) = (b, h, s, d_k )
						 the output is (b, s, d_model) when reshaped cuz h * d_k = d_model*/

                        matrixmult(output, p->W_O, output, total, d_model, d_model);
                        bias_summ_krln<<<(n_qkv+threads-1)/threads, threads>>>(output, p->b_O, total, d_model);/*project the concated multi-head outpt thru W_O -a d_model * d_model learned mixing mtx.
						this allows the model to learn how to combine info fro mdiff heads., use diff buffer for the matrixmult() to prevent race.*/


 } // issue in variable passing in bias addition kernel (total) which is "int" but being passed as "const float*" --> removed the imput redundancy as there were 4 arguments and we passed offset argument in the add summation kernel.



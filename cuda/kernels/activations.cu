// actual activationnnnn baby.

#define GeLU_coeff 0.044715f
#define SQRT_2_pi 0.7978845608f //sqrt(2/pi)

__device__ inline float GeLU_val(float x){
    if (x > 10.0f) return x;
    if (x < -10.0f) return 0.0f;
    float t = SQRT_2_pi * (x + GeLU_coeff * x * x * x);
    return 0.5f * x * (1.0f + tanhf(t)); //tanhf(single precision tanh) faster than tanh((double)causes implicit widening to double precision) on GPU.
}

__device__ inline float GeLU_deriv(float x){
    if (x > 10.0f) return 1.0f;
    if (x < -10.0f) return 0.0f;
    float t = SQRT_2_pi * (x + GeLU_coeff * x * x * x);
    float tanh_t = tanhf(t);
    float sech2_t = 1.0f - tanh_t * tanh_t; //sech^2(t) = 1 - tanh^2(t)
    float dt_dx = SQRT_2_pi * (1.0f + 3.0f * GeLU_coeff * x * x);
    return 0.5f * (1.0f + tanh_t) + 0.5f * x * sech2_t * dt_dx;
} // d(GeLU)/dx = 0.5 * (1 + tanh(t)) + 0.5 * x * sech^2(t) * dt/dx.

__global__ void GeLU_fwd_kernel(float* output, const float* in, int n) {

    int i = blockIdx.x * blockDim.x + threadIdx.x; //1d index formula convverts 2D grid position(block, thread) into a single flat index. => eg. 2B and 5Th, blockDim 256 -> i = 2*256 + 5 = 517.
    if (i < n){
       output[i] = GeLU_val(in[i]);
    }

}

__global__ void GeLU_bwd_kernel(float* d_in, const float* d_out, const float* input, int n) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
       d_in[i] = d_out[i] * GeLU_deriv(input[i]);
    }

}

__global__ void ReLU_fwd_kernel(float* output, const float* input, int n) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
      output[i] = fmaxf(0.0f, input[i]);
    }

}/* ReLU(x) = x>=0 => 1
              x<=0 => 0
 d(ReLU)/dx = x>=0 => 1 */

__global__ void ReLU_bwd_kernel(float* d_input, const float* d_output, const float* input, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        d_input[i] = (input[i] > 0.0f) ? d_output[i] : 0.0f;
    }
}

void GeLU_forward(float* output, const float* input, int n) {

    int threads = 256;
    GeLU_fwd_kernel<<<(n + threads - 1) / threads, threads>>>(output, input, n); //ceiling division inthe first kernel dimension equivalent to ceil(n/256).
    //ensure enough bloacks to cover all n elements, even if not divisible by 256.

}

void GeLU_backward(float* d_input, const float* d_output, const float* input, int n) {
    int threads = 256;
    GeLU_bwd_kernel<<<(n + threads - 1) / threads, threads>>>(d_input, d_output, input, n);
}

void ReLU_forward(float* output, const float* input, int n) {
    int threads = 256;
    ReLU_fwd_kernel<<<(n + threads - 1) / threads, threads>>>(output, input, n);
}

void ReLU_backward(float* d_input, const float* d_output, const float* input, int n) {
    int threads = 256;
    ReLU_bwd_kernel<<<(n + threads - 1) / threads, threads>>>(d_input, d_output, input, n);
}
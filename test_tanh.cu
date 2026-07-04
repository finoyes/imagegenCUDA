#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
__global__ void ker(float* out) {
    float x = 1e15f;
    float t = 0.8f * (x + 0.044f * x * x * x);
    float tanh_t = tanhf(t);
    float sech2_t = 1.0f - tanh_t * tanh_t;
    float dt_dx = 0.8f * (1.0f + 3.0f * 0.044f * x * x);
    *out = 0.5f * (1.0f + tanh_t) + 0.5f * x * sech2_t * dt_dx;
}
int main() {
    float h=0, *d; cudaMalloc(&d, 4);
    ker<<<1,1>>>(d); cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost);
    printf("%f\n", h);
    return 0;
}

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
__global__ void ker(float* out) { *out = sqrtf(-1.0f); }
int main() {
    float h=0, *d; cudaMalloc(&d, 4);
    ker<<<1,1>>>(d); cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost);
    printf("%x\n", *(unsigned int*)&h);
    return 0;
}

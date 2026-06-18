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
// evrey cudaMalloc call is expensive, it syncs the gpu and may involve a kernel to initialize memory. in a training loop where you allocatio temp buffers for intermidiate activationss, doing this per-step kills performance.
// to solve this we use arena allocator , allocate one large GPU buffer at startup, then sub-allocate from it by just bumping a pointer. freeing is O(1) - just reset the pointer to the arena start. since training steps have predictable memory layouts, you allocate the same set of tensors every step.

# pragma once
#include <cuda_runtime.h>
#include <cstddef>

struct GpuArena{
    float* base;// start of GPU allocation
    size_t capacity;// total bytes for alloaction
    size_t used;// bytes under use
};

GpuArena arena_create(size_t bytes);
void     arena_destroy(GpuArena* a);
float*   arena_alloc(GpuArena* a, size_t n_floats);
void     arena_reset(GpuArena* a);   // free all sub-allocations at once
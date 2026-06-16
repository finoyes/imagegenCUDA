#include "scheduler.h"
#include <cuda_runtime.h>
#include "optimizer.cuh"
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

// problems
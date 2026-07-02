
#pragma once

struct LRScheduler {

    float max_lr;
    float min_lr;
    int warmup_steps;
    int total_steps;

};

LRScheduler scheduler_create(float max_lr, float min_lr, int warmup_step, int total_steps);

float scheduler_get_lr(const LRScheduler* s, int step);

void scheduler_step(const LRScheduler* s, struct AdamWOptimizer* opt, int step);


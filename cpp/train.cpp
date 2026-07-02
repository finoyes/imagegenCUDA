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

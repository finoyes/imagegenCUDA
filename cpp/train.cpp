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
#include <filesystem>
#include <stdexcept>
#include <vector>
#include <string>
#include <cmath>

namespace fs = std::filesystem;

// Walk up the directory tree (up to max_levels) to find a relative path.
static std::string find_path(const std::string& rel, int max_levels = 5) {
    fs::path candidate = rel;
    for (int i = 0; i <= max_levels; i++) {
        if (fs::exists(candidate))
            return candidate.string();
        candidate = fs::path("..") / candidate;
    }
    throw std::runtime_error(
        "Cannot find '" + rel + "' — run train.exe from the project root "
        "or any subdirectory up to " + std::to_string(max_levels) + " levels deep.");
}

static void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

static float* alloc_device_floats(size_t count, float fill_value = 0.0f) {
    float* ptr = nullptr;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(float)), "cudaMalloc");
    if (fill_value == 0.0f) {
        cuda_check(cudaMemset(ptr, 0, count * sizeof(float)), "cudaMemset");
    } else {
        std::vector<float> host(count, fill_value);
        cuda_check(cudaMemcpy(ptr, host.data(), count * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");
    }
    return ptr;
}

// Xavier/Glorot uniform initialisation on CPU then copy to GPU.
// Keeps initial activations in a reasonable range for d_model=256.
static void xavier_init(float* d_ptr, int fan_in, int fan_out) {
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    int n = fan_in * fan_out;
    std::vector<float> h(n);
    for (int i = 0; i < n; i++) {
        float r = (float)rand() / (float)RAND_MAX;  // [0,1]
        h[i] = (2.0f * r - 1.0f) * limit;           // [-limit, limit]
    }
    cudaMemcpy(d_ptr, h.data(), n * sizeof(float), cudaMemcpyHostToDevice);
}

static void init_encoder_params(EncoderParams& enc, const EncoderConfig& enc_cfg) {
    enc.patch_proj_W   = alloc_device_floats((size_t)enc_cfg.patch_dim * enc_cfg.d_model);
    enc.patch_proj_b   = alloc_device_floats(enc_cfg.d_model);
    enc.pos_embedding  = alloc_device_floats((size_t)enc_cfg.num_patches * enc_cfg.d_model);
    enc.cls_token      = alloc_device_floats(enc_cfg.d_model);
    enc.d_patch_proj_W = alloc_device_floats((size_t)enc_cfg.patch_dim * enc_cfg.d_model);
    enc.d_patch_proj_b = alloc_device_floats(enc_cfg.d_model);
    enc.d_pos_embedding= alloc_device_floats((size_t)enc_cfg.num_patches * enc_cfg.d_model);
    xavier_init(enc.patch_proj_W, enc_cfg.patch_dim, enc_cfg.d_model);
    encoder_init(&enc, &enc_cfg);
}

static void init_decoder_params(DecoderParams& dec, const EncoderConfig& enc_cfg) {
    dec.proj_W   = alloc_device_floats((size_t)enc_cfg.d_model * enc_cfg.patch_dim);
    dec.proj_b   = alloc_device_floats(enc_cfg.patch_dim);
    dec.d_proj_W = alloc_device_floats((size_t)enc_cfg.d_model * enc_cfg.patch_dim);
    dec.d_proj_b = alloc_device_floats(enc_cfg.patch_dim);
    xavier_init(dec.proj_W, enc_cfg.d_model, enc_cfg.patch_dim);
}

static void init_transformer_params(TransformerParams& tr, const ModelConfig& cfg) {
    tr.num_layers = cfg.num_layers;
    tr.d_model    = cfg.d_model;
    tr.num_heads  = cfg.num_heads;
    tr.d_ff       = cfg.d_ff;
    tr.blocks     = new TransformerBlockParams[tr.num_layers];

    for (int l = 0; l < tr.num_layers; l++) {
        TransformerBlockParams& bp = tr.blocks[l];
        // Attention weights
        bp.attn.W_Q = alloc_device_floats((size_t)cfg.d_model * cfg.d_model);
        bp.attn.W_K = alloc_device_floats((size_t)cfg.d_model * cfg.d_model);
        bp.attn.W_V = alloc_device_floats((size_t)cfg.d_model * cfg.d_model);
        bp.attn.W_O = alloc_device_floats((size_t)cfg.d_model * cfg.d_model);
        bp.attn.b_Q = alloc_device_floats(cfg.d_model);
        bp.attn.b_K = alloc_device_floats(cfg.d_model);
        bp.attn.b_V = alloc_device_floats(cfg.d_model);
        bp.attn.b_O = alloc_device_floats(cfg.d_model);
        // Attention gradients
        bp.attn.dW_Q = alloc_device_floats((size_t)cfg.d_model * cfg.d_model);
        bp.attn.dW_K = alloc_device_floats((size_t)cfg.d_model * cfg.d_model);
        bp.attn.dW_V = alloc_device_floats((size_t)cfg.d_model * cfg.d_model);
        bp.attn.dW_O = alloc_device_floats((size_t)cfg.d_model * cfg.d_model);
        bp.attn.db_Q = alloc_device_floats(cfg.d_model);
        bp.attn.db_K = alloc_device_floats(cfg.d_model);
        bp.attn.db_V = alloc_device_floats(cfg.d_model);
        bp.attn.db_O = alloc_device_floats(cfg.d_model);
        // Xavier init for projection matrices
        xavier_init(bp.attn.W_Q, cfg.d_model, cfg.d_model);
        xavier_init(bp.attn.W_K, cfg.d_model, cfg.d_model);
        xavier_init(bp.attn.W_V, cfg.d_model, cfg.d_model);
        xavier_init(bp.attn.W_O, cfg.d_model, cfg.d_model);

        // FFN weights
        bp.ffn.W1  = alloc_device_floats((size_t)cfg.d_model * cfg.d_ff);
        bp.ffn.b1  = alloc_device_floats(cfg.d_ff);
        bp.ffn.W2  = alloc_device_floats((size_t)cfg.d_ff * cfg.d_model);
        bp.ffn.b2  = alloc_device_floats(cfg.d_model);
        bp.ffn.dW1 = alloc_device_floats((size_t)cfg.d_model * cfg.d_ff);
        bp.ffn.db1 = alloc_device_floats(cfg.d_ff);
        bp.ffn.dW2 = alloc_device_floats((size_t)cfg.d_ff * cfg.d_model);
        bp.ffn.db2 = alloc_device_floats(cfg.d_model);
        xavier_init(bp.ffn.W1, cfg.d_model, cfg.d_ff);
        xavier_init(bp.ffn.W2, cfg.d_ff,    cfg.d_model);

        // Layer norm (gamma=1, beta=0 by default)
        bp.ln1_gamma   = alloc_device_floats(cfg.d_model, 1.0f);
        bp.ln1_beta    = alloc_device_floats(cfg.d_model);
        bp.ln2_gamma   = alloc_device_floats(cfg.d_model, 1.0f);
        bp.ln2_beta    = alloc_device_floats(cfg.d_model);
        bp.d_ln1_gamma = alloc_device_floats(cfg.d_model);
        bp.d_ln1_beta  = alloc_device_floats(cfg.d_model);
        bp.d_ln2_gamma = alloc_device_floats(cfg.d_model);
        bp.d_ln2_beta  = alloc_device_floats(cfg.d_model);
    }
}

static void init_transformer_cache(TransformerCache& cache, const ModelConfig& cfg,
                                   int seq_len)
{
    int L  = cfg.num_layers;
    int B  = cfg.batch_size;
    int D  = cfg.d_model;
    int dff = cfg.d_ff;
    int H  = cfg.num_heads;
    int dk = D / H;

    cache.ln1_out      = new float*[L];
    cache.attn_out     = new float*[L];
    cache.ln2_out      = new float*[L];
    cache.ffn_mid      = new float*[L];
    cache.ffn_pre_gelu = new float*[L];
    cache.x_after_attn = new float*[L];
    cache.attn_caches  = new AttentionCache[L];

    for (int l = 0; l < L; l++) {
        cache.ln1_out[l]      = alloc_device_floats((size_t)B * seq_len * D);
        cache.attn_out[l]     = alloc_device_floats((size_t)B * seq_len * D);
        cache.ln2_out[l]      = alloc_device_floats((size_t)B * seq_len * D);
        cache.ffn_mid[l]      = alloc_device_floats((size_t)B * seq_len * dff);
        cache.ffn_pre_gelu[l] = alloc_device_floats((size_t)B * seq_len * dff);
        cache.x_after_attn[l] = alloc_device_floats((size_t)B * seq_len * D);

        AttentionCache& ac = cache.attn_caches[l];
        ac.Q       = alloc_device_floats((size_t)B * seq_len * D);
        ac.K       = alloc_device_floats((size_t)B * seq_len * D);
        ac.V       = alloc_device_floats((size_t)B * seq_len * D);
        ac.Q_h     = alloc_device_floats((size_t)B * H * seq_len * dk);
        ac.K_h     = alloc_device_floats((size_t)B * H * seq_len * dk);
        ac.V_h     = alloc_device_floats((size_t)B * H * seq_len * dk);
        ac.Kt_h    = alloc_device_floats((size_t)B * H * dk * seq_len);
        ac.scores  = alloc_device_floats((size_t)B * H * seq_len * seq_len);
        ac.input   = alloc_device_floats((size_t)B * seq_len * D);
        ac.proj_in = alloc_device_floats((size_t)B * seq_len * D);
    }
}

// Register all trainable parameters with the optimizer.
// Each call to adamw_add_param records (param_ptr, grad_ptr, size).
static void register_optimizer_params(AdamWOptimizer& opt,
                                      EncoderParams& enc,
                                      TransformerParams& tr,
                                      DecoderParams& dec,
                                      const EncoderConfig& enc_cfg,
                                      const ModelConfig& cfg)
{
    // ── Encoder ──────────────────────────────────────────────────────────────
    adamw_add_param(&opt, enc.patch_proj_W, enc.d_patch_proj_W,
                    enc_cfg.patch_dim * enc_cfg.d_model);

    // ── Transformer ───────────────────────────────────────────────────────────
    for (int l = 0; l < tr.num_layers; l++) {
        TransformerBlockParams& bp = tr.blocks[l];
        int DD = cfg.d_model * cfg.d_model;
        int D  = cfg.d_model;
        int Df = cfg.d_model * cfg.d_ff;
        int dff = cfg.d_ff;

        adamw_add_param(&opt, bp.attn.W_Q, bp.attn.dW_Q, DD);
        adamw_add_param(&opt, bp.attn.W_K, bp.attn.dW_K, DD);
        adamw_add_param(&opt, bp.attn.W_V, bp.attn.dW_V, DD);
        adamw_add_param(&opt, bp.attn.W_O, bp.attn.dW_O, DD);
        adamw_add_param(&opt, bp.attn.b_Q, bp.attn.db_Q, D);
        adamw_add_param(&opt, bp.attn.b_K, bp.attn.db_K, D);
        adamw_add_param(&opt, bp.attn.b_V, bp.attn.db_V, D);
        adamw_add_param(&opt, bp.attn.b_O, bp.attn.db_O, D);

        adamw_add_param(&opt, bp.ffn.W1, bp.ffn.dW1, Df);
        adamw_add_param(&opt, bp.ffn.b1, bp.ffn.db1, dff);
        adamw_add_param(&opt, bp.ffn.W2, bp.ffn.dW2, Df);
        adamw_add_param(&opt, bp.ffn.b2, bp.ffn.db2, D);

        adamw_add_param(&opt, bp.ln1_gamma, bp.d_ln1_gamma, D);
        adamw_add_param(&opt, bp.ln1_beta,  bp.d_ln1_beta,  D);
        adamw_add_param(&opt, bp.ln2_gamma, bp.d_ln2_gamma, D);
        adamw_add_param(&opt, bp.ln2_beta,  bp.d_ln2_beta,  D);
    }

    // ── Decoder ───────────────────────────────────────────────────────────────
    adamw_add_param(&opt, dec.proj_W, dec.d_proj_W,
                    enc_cfg.d_model * enc_cfg.patch_dim);
}

int main(int argc, char** argv) {
    try {
        // ── Load config ───────────────────────────────────────────────────────
        std::string config_path = find_path("config.json");
        printf("[train] Using config: %s\n", config_path.c_str());
        std::ifstream f(config_path);
        if (!f.is_open()) {
            fprintf(stderr, "Unable to open config file: %s\n", config_path.c_str());
            return 1;
        }
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

        int   total_steps    = cfg_json["total_steps"];
        int   warmup_steps   = cfg_json["warmup_steps"];
        float max_lr         = cfg_json["max_lr"];
        float min_lr         = cfg_json["min_lr"];
        float weight_decay   = cfg_json["weight_decay"];
        int   save_every     = cfg_json["save_every"];

        EncoderConfig enc_cfg;
        enc_cfg.image_size   = cfg.image_size;
        enc_cfg.patch_size   = cfg.patch_size;
        enc_cfg.channels     = cfg.channels;
        enc_cfg.d_model      = cfg.d_model;
        enc_cfg.num_patches  = (cfg.image_size / cfg.patch_size) *
                               (cfg.image_size / cfg.patch_size);
        enc_cfg.patch_dim    = cfg.patch_size * cfg.patch_size * cfg.channels;

        // ── Allocate all GPU parameters ───────────────────────────────────────
        EncoderParams     enc_params  = {};
        TransformerParams tr_params   = {};
        DecoderParams     dec_params  = {};
        init_encoder_params(enc_params, enc_cfg);
        init_transformer_params(tr_params, cfg);
        init_decoder_params(dec_params, enc_cfg);

        // ── Optimizer ─────────────────────────────────────────────────────────
        AdamWOptimizer opt = adamw_create(max_lr, 0.9f, 0.999f, 1e-8f, weight_decay);
        register_optimizer_params(opt, enc_params, tr_params, dec_params, enc_cfg, cfg);
        printf("[train] Optimizer registered %d parameter tensors\n", (int)opt.params.size());

        LRScheduler scheduler = scheduler_create(max_lr, min_lr, warmup_steps, total_steps);

        // ── Data ──────────────────────────────────────────────────────────────
        std::string data_dir       = find_path("data/processed");
        std::string log_dir        = find_path("logs");
        std::string checkpoint_dir = find_path("checkpoints");

        printf("[train] data dir      : %s\n", data_dir.c_str());
        printf("[train] log dir       : %s\n", log_dir.c_str());
        printf("[train] checkpoint dir: %s\n", checkpoint_dir.c_str());
        fflush(stdout);

        DataLoader loader(data_dir, cfg.batch_size, cfg.image_size, cfg.channels);
        const int batches_per_epoch = loader.num_batches();
        if (batches_per_epoch == 0) {
            fprintf(stderr, "[train] ERROR: num_batches() == 0. "
                    "dataset has fewer images than batch_size (%d). Aborting.\n",
                    cfg.batch_size);
            return 1;
        }
        printf("[train] %d files found → %d batches/epoch\n",
               (int)(batches_per_epoch * cfg.batch_size), batches_per_epoch);
        fflush(stdout);

        Logger logger(log_dir, save_every);

        // ── Activation buffers (GPU) ──────────────────────────────────────────
        int seq_len = enc_cfg.num_patches;
        int N  = cfg.batch_size * seq_len;
        int D  = cfg.d_model, C = cfg.channels, H = cfg.image_size;
        float* d_tokens    = alloc_device_floats((size_t)N * D);
        float* d_tr_out    = alloc_device_floats((size_t)N * D);
        float* d_recon     = alloc_device_floats((size_t)cfg.batch_size * C * H * H);
        float* d_loss      = alloc_device_floats(1);
        float* d_loss_grad = alloc_device_floats((size_t)cfg.batch_size * C * H * H);
        // Gradient buffers for the backward chain
        float* d_tr_grad   = alloc_device_floats((size_t)N * D);   // dL/d(tr_out)
        float* d_enc_grad  = alloc_device_floats((size_t)N * D);   // dL/d(tokens)
        float* d_img_grad  = alloc_device_floats((size_t)cfg.batch_size * C * H * H);

        TransformerCache tr_cache = {};
        init_transformer_cache(tr_cache, cfg, seq_len);

        // ── Resume from checkpoint ────────────────────────────────────────────
        int start_step = 0;
        const std::string latest_checkpoint = checkpoint_dir + "/latest.bin";
        if (checkpoint_exists(latest_checkpoint)) {
            start_step = checkpoint_load(latest_checkpoint,
                                         &enc_params, &tr_params, &dec_params, &cfg);
            printf("[train] Resuming from checkpoint at step %d\n", start_step);
        } else {
            printf("[train] No checkpoint found; starting from scratch.\n");
        }
        fflush(stdout);

        // ── Training loop ─────────────────────────────────────────────────────
        printf("[train] Starting training: total_steps=%d, batch_size=%d, lr=%.2e\n",
               total_steps, cfg.batch_size, (double)max_lr);
        fflush(stdout);

        int step = start_step;
        for (int epoch = 0; step < total_steps; epoch++) {
            for (int b = 0; b < batches_per_epoch && step < total_steps; b++, step++) {

                float* d_images = loader.next_batch();

                // ── Forward pass ──────────────────────────────────────────────
                encoder_forward(d_tokens, d_images, &enc_params, &enc_cfg, cfg.batch_size);
                transformer_forward(d_tr_out, &tr_cache, d_tokens, &tr_params,
                                    cfg.batch_size, seq_len);
                decoder_forward(d_recon, d_tr_out, &dec_params, &enc_cfg, cfg.batch_size);

                // ── Loss ──────────────────────────────────────────────────────
                mse_loss(d_loss, d_loss_grad, d_recon, d_images,
                         cfg.batch_size * C * H * H);

                float loss_val = 0.0f;
                cuda_check(cudaMemcpy(&loss_val, d_loss, sizeof(float),
                                      cudaMemcpyDeviceToHost), "cudaMemcpy loss");

                // ── Backward pass ─────────────────────────────────────────────
                // Chain: decoder ← transformer ← encoder
                decoder_backward(d_tr_grad, &dec_params,
                                 d_loss_grad, d_recon,
                                 d_tr_out, &enc_cfg, cfg.batch_size);

                transformer_backward_full(d_enc_grad, &tr_params,
                                         d_tr_grad, &tr_cache,
                                         cfg.batch_size, seq_len);

                encoder_backward(d_img_grad, &enc_params,
                                 d_enc_grad, d_images, &enc_cfg, cfg.batch_size);

                // ── Optimizer step ────────────────────────────────────────────
                clip_grad_norm(&opt, 1.0f);
                adamw_step(&opt);
                adamw_zero_grad(&opt);
                scheduler_step(&scheduler, &opt, step);

                // ── Gradient norm debug (first 10 steps only) ─────────────────
                if (step < 10) {
                    int gsz = enc_cfg.patch_dim * enc_cfg.d_model;
                    std::vector<float> h_grad(gsz);
                    cudaMemcpy(h_grad.data(), enc_params.d_patch_proj_W,
                               gsz * sizeof(float), cudaMemcpyDeviceToHost);
                    float gnorm = 0.0f;
                    for (float g : h_grad) gnorm += g * g;
                    printf("[debug] step %d  loss=%.4f  enc_patch_proj grad_norm=%.6f\n",
                           step, loss_val, sqrtf(gnorm));
                    fflush(stdout);
                }

                logger.log_loss(step, epoch, loss_val, opt.lr);
                logger.save_samples(step, d_recon, d_images,
                                    cfg.batch_size, C, H);

                if (step > 0 && step % save_every == 0) {
                    char ckpt_path[256];
                    snprintf(ckpt_path, 256, "%s/step_%06d.bin",
                             checkpoint_dir.c_str(), step);
                    checkpoint_save(ckpt_path, &enc_params, &tr_params, &dec_params, cfg, step);
                    checkpoint_save(latest_checkpoint.c_str(), &enc_params, &tr_params,
                                    &dec_params, cfg, step);
                    printf("[train] Checkpoint saved at step %d\n", step);
                    fflush(stdout);
                }
            }
        }

        printf("Training complete.\n");
        return 0;
    } catch (const std::exception& ex) {
        fprintf(stderr, "Training failed: %s\n", ex.what());
        return 1;
    }
}

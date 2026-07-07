#include "checkpoint.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cuda_runtime.h>

#define MAGIC   0x54524E47u  // "TRNG"
#define VERSION 1

// Helper: copy GPU tensor to CPU buffer and write to file
static void write_tensor(FILE* f, const float* d_ptr, int n) {
    std::vector<float> buf(n);
    cudaMemcpy(buf.data(), d_ptr, n * sizeof(float), cudaMemcpyDeviceToHost);
    fwrite(buf.data(), sizeof(float), n, f);
}

// Helper: read from file into GPU tensor
static void read_tensor(FILE* f, float* d_ptr, int n) {
    std::vector<float> buf(n);
    fread(buf.data(), sizeof(float), n, f);
    cudaMemcpy(d_ptr, buf.data(), n * sizeof(float), cudaMemcpyHostToDevice);
}

void checkpoint_save(
    const std::string& path,
    const EncoderParams* enc, const TransformerParams* tr,
    const DecoderParams* dec, const ModelConfig& cfg, int step,
    const AdamWOptimizer* opt)
{
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return; }

    // Header
    uint32_t magic = MAGIC;
    int version = VERSION;
    fwrite(&magic,   sizeof(uint32_t), 1, f);
    fwrite(&version, sizeof(int),      1, f);
    fwrite(&cfg,     sizeof(ModelConfig), 1, f);
    fwrite(&step,    sizeof(int),      1, f);

    int D  = cfg.d_model, PD = cfg.patch_size * cfg.patch_size * cfg.channels;
    int NP = (cfg.image_size / cfg.patch_size) * (cfg.image_size / cfg.patch_size);

    // Encoder weights
    write_tensor(f, enc->patch_proj_W, PD * D);
    write_tensor(f, enc->patch_proj_b, D);
    write_tensor(f, enc->pos_embedding, NP * D);

    // Transformer weights (all layers)
    for (int l = 0; l < tr->num_layers; l++) {
        const TransformerBlockParams& bp = tr->blocks[l];
        write_tensor(f, bp.attn.W_Q, D * D);
        write_tensor(f, bp.attn.W_K, D * D);
        write_tensor(f, bp.attn.W_V, D * D);
        write_tensor(f, bp.attn.W_O, D * D);
        write_tensor(f, bp.ffn.W1, D * cfg.d_ff);
        write_tensor(f, bp.ffn.b1, cfg.d_ff);
        write_tensor(f, bp.ffn.W2, cfg.d_ff * D);
        write_tensor(f, bp.ffn.b2, D);
        write_tensor(f, bp.ln1_gamma, D);
        write_tensor(f, bp.ln1_beta, D);
        write_tensor(f, bp.ln2_gamma, D);
        write_tensor(f, bp.ln2_beta, D);
    }

    // Decoder weights
    write_tensor(f, dec->proj_W, D * PD);
    write_tensor(f, dec->proj_b, PD);

    // ── Optimizer state ───────────────────────────────────────────────────────
    // Written only when opt != nullptr. Tag lets checkpoint_load detect whether
    // optimizer state is present even on old checkpoints that lack it.
    uint32_t opt_tag = (opt != nullptr) ? 0x4F505420u : 0u;  // "OPT " or 0
    fwrite(&opt_tag, sizeof(uint32_t), 1, f);

    if (opt != nullptr) {
        // Step counter — critical for AdamW bias correction (bc1, bc2)
        fwrite(&opt->step, sizeof(int), 1, f);

        // Number of parameter tensors (sanity check on load)
        int num_tensors = (int)opt->states.size();
        fwrite(&num_tensors, sizeof(int), 1, f);

        // m and v buffers for every registered tensor
        for (int i = 0; i < num_tensors; i++) {
            write_tensor(f, opt->states[i].m, opt->sizes[i]);
            write_tensor(f, opt->states[i].v, opt->sizes[i]);
        }
    }

    fclose(f);
    printf("Saved checkpoint: %s (step %d, optimizer %s)\n",
           path.c_str(), step, opt ? "included" : "not saved");
}

int checkpoint_load(
    const std::string& path,
    EncoderParams* enc, TransformerParams* tr,
    DecoderParams* dec, ModelConfig* cfg,
    AdamWOptimizer* opt)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return -1; }

    uint32_t magic; int version;
    fread(&magic,   sizeof(uint32_t),   1, f);
    fread(&version, sizeof(int),        1, f);
    fread(cfg,      sizeof(ModelConfig),1, f);
    int step; fread(&step, sizeof(int), 1, f);

    if (magic != MAGIC) { fprintf(stderr, "bad checkpoint magic\n"); fclose(f); return -1; }

    int D = cfg->d_model, PD = cfg->patch_size * cfg->patch_size * cfg->channels;
    int NP = (cfg->image_size / cfg->patch_size) * (cfg->image_size / cfg->patch_size);

    read_tensor(f, enc->patch_proj_W, PD * D);
    read_tensor(f, enc->patch_proj_b, D);
    read_tensor(f, enc->pos_embedding, NP * D);

    for (int l = 0; l < tr->num_layers; l++) {
        TransformerBlockParams& bp = tr->blocks[l];
        read_tensor(f, bp.attn.W_Q, D * D);
        read_tensor(f, bp.attn.W_K, D * D);
        read_tensor(f, bp.attn.W_V, D * D);
        read_tensor(f, bp.attn.W_O, D * D);
        read_tensor(f, bp.ffn.W1, D * cfg->d_ff);
        read_tensor(f, bp.ffn.b1, cfg->d_ff);
        read_tensor(f, bp.ffn.W2, cfg->d_ff * D);
        read_tensor(f, bp.ffn.b2, D);
        read_tensor(f, bp.ln1_gamma, D);
        read_tensor(f, bp.ln1_beta, D);
        read_tensor(f, bp.ln2_gamma, D);
        read_tensor(f, bp.ln2_beta, D);
    }

    read_tensor(f, dec->proj_W, D * PD);
    read_tensor(f, dec->proj_b, PD);

    // ── Optimizer state ───────────────────────────────────────────────────────
    uint32_t opt_tag = 0;
    if (fread(&opt_tag, sizeof(uint32_t), 1, f) == 1 && opt_tag == 0x4F505420u) {
        // Optimizer state present in this checkpoint
        int saved_step = 0;
        fread(&saved_step, sizeof(int), 1, f);

        int num_tensors = 0;
        fread(&num_tensors, sizeof(int), 1, f);

        if (opt != nullptr) {
            opt->step = saved_step;

            if (num_tensors == (int)opt->states.size()) {
                for (int i = 0; i < num_tensors; i++) {
                    read_tensor(f, opt->states[i].m, opt->sizes[i]);
                    read_tensor(f, opt->states[i].v, opt->sizes[i]);
                }
                printf("Loaded optimizer state (step=%d, tensors=%d)\n",
                       saved_step, num_tensors);
            } else {
                fprintf(stderr,
                    "[checkpoint] WARNING: optimizer tensor count mismatch "
                    "(file=%d, current=%d). Skipping optimizer state restore.\n",
                    num_tensors, (int)opt->states.size());
                // Still restore the step counter — that's the most critical part
            }
        }
    } else {
        // Old checkpoint without optimizer state
        if (opt != nullptr) {
            // CRITICAL: restore step counter from the model step so AdamW
            // bias correction (bc1 = 1 - beta1^step) doesn't divide by zero.
            opt->step = step;
            fprintf(stderr,
                "[checkpoint] WARNING: no optimizer state in checkpoint. "
                "Setting opt.step=%d to avoid bias-correction division by zero. "
                "Momentum buffers (m, v) are reset — expect a brief loss spike.\n",
                step);
        }
    }

    fclose(f);
    printf("Loaded checkpoint: %s (step %d)\n", path.c_str(), step);
    return step;
}

bool checkpoint_exists(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (f) { fclose(f); return true; }
    return false;
}

//explain the file.

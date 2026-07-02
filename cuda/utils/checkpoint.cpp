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
    const DecoderParams* dec, const ModelConfig& cfg, int step)
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
    write_tensor(f, enc->patch_projection_w, PD * D);
    write_tensor(f, enc->patch_projection_b, D);
    write_tensor(f, enc->pos_embedding, NP * D);

    // Transformer weights (all layers)
    for (int l = 0; l < tr->num_layers; l++) {
        const TransformerBlockParams& bp = tr->blocks[l];
        write_tensor(f, bp.attn.W_Q, D * D);
        write_tensor(f, bp.attn.W_K, D * D);
        write_tensor(f, bp.attn.W_V, D * D);
        write_tensor(f, bp.attn.W_O, D * D);
        // biases, layer norm params, FFN weights...
    }

    // Decoder weights
    write_tensor(f, dec->proj_W, D * PD);
    write_tensor(f, dec->proj_b, PD);

    fclose(f);
    printf("Saved checkpoint: %s (step %d)\n", path.c_str(), step);
}

int checkpoint_load(
    const std::string& path,
    EncoderParams* enc, TransformerParams* tr,
    DecoderParams* dec, ModelConfig* cfg)
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

    read_tensor(f, enc->patch_projection_w, PD * D);
    read_tensor(f, enc->patch_projection_b, D);
    read_tensor(f, enc->pos_embedding, NP * D);

    for (int l = 0; l < tr->num_layers; l++) {
        TransformerBlockParams& bp = tr->blocks[l];
        read_tensor(f, bp.attn.W_Q, D * D);
        read_tensor(f, bp.attn.W_K, D * D);
        read_tensor(f, bp.attn.W_V, D * D);
        read_tensor(f, bp.attn.W_O, D * D);
    }

    read_tensor(f, dec->proj_W, D * PD);
    read_tensor(f, dec->proj_b, PD);

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

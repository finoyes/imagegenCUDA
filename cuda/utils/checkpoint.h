// checkpointing saves all learned weight tensors to disk so you can, resume training after interreption, load weights for interference, compare snapshots across training

#pragma once
#include <string>
#include "../model/transformer.cuh"
#include "../model/encoder.cuh"
#include "../model/decoder.cuh"
#include "../training/optimizer.cuh"

struct ModelConfig {
    int image_size, patch_size, channels;
    int d_model, num_heads, num_layers, d_ff;
    int batch_size;
};

// Save all model parameters + optimizer state (step, m, v) to a binary file.
void checkpoint_save(
    const std::string& path,
    const EncoderParams* enc,
    const TransformerParams* tr,
    const DecoderParams* dec,
    const ModelConfig& cfg,
    int step,
    const AdamWOptimizer* opt = nullptr   // pass nullptr to skip optimizer state
);

// Load model parameters + optimizer state from file into already-allocated GPU buffers.
// Returns the step number stored in the checkpoint.
int checkpoint_load(
    const std::string& path,
    EncoderParams* enc,
    TransformerParams* tr,
    DecoderParams* dec,
    ModelConfig* cfg,
    AdamWOptimizer* opt = nullptr         // pass nullptr to skip optimizer state
);

bool checkpoint_exists(const std::string& path);
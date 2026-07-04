#include <stdio.h>
#include "../cuda/utils/checkpoint.h"
#include "../cuda/model/encoder.cuh"
#include "../cuda/model/transformer.cuh"
#include "../cuda/model/decoder.cuh"
#include <vector>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../vendor/stb_image.h"
#include "../vendor/stb_image_write.h"
#include "../vendor/stb_image_resize.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: generate <checkpoint.bin> <input_image.jpg> [output.png]\n");
        return 1;
    }

    ModelConfig cfg;
    // We need to read the config first to allocate memory
    FILE* f = fopen(argv[1], "rb");
    if (!f) return 1;
    uint32_t magic; int version;
    fread(&magic, sizeof(uint32_t), 1, f);
    fread(&version, sizeof(int), 1, f);
    fread(&cfg, sizeof(ModelConfig), 1, f);
    fclose(f);

    int num_patches = (cfg.image_size / cfg.patch_size) * (cfg.image_size / cfg.patch_size);
    int PD = cfg.patch_size * cfg.patch_size * cfg.channels;
    
    EncoderParams  enc = {};
    cudaMalloc(&enc.patch_proj_W, PD * cfg.d_model * sizeof(float));
    cudaMalloc(&enc.patch_proj_b, cfg.d_model * sizeof(float));
    cudaMalloc(&enc.pos_embedding, num_patches * cfg.d_model * sizeof(float));
    
    TransformerParams tr = {};
    tr.num_layers = cfg.num_layers;
    tr.d_model = cfg.d_model;
    tr.num_heads = cfg.num_heads;
    tr.d_ff = cfg.d_ff;
    tr.blocks = (TransformerBlockParams*)malloc(cfg.num_layers * sizeof(TransformerBlockParams));
    for (int i = 0; i < cfg.num_layers; i++) {
        cudaMalloc(&tr.blocks[i].attn.W_Q, cfg.d_model * cfg.d_model * sizeof(float));
        cudaMalloc(&tr.blocks[i].attn.W_K, cfg.d_model * cfg.d_model * sizeof(float));
        cudaMalloc(&tr.blocks[i].attn.W_V, cfg.d_model * cfg.d_model * sizeof(float));
        cudaMalloc(&tr.blocks[i].attn.W_O, cfg.d_model * cfg.d_model * sizeof(float));
        cudaMalloc(&tr.blocks[i].ffn.W1, cfg.d_model * cfg.d_ff * sizeof(float));
        cudaMalloc(&tr.blocks[i].ffn.b1, cfg.d_ff * sizeof(float));
        cudaMalloc(&tr.blocks[i].ffn.W2, cfg.d_ff * cfg.d_model * sizeof(float));
        cudaMalloc(&tr.blocks[i].ffn.b2, cfg.d_model * sizeof(float));
        cudaMalloc(&tr.blocks[i].ln1_gamma, cfg.d_model * sizeof(float));
        cudaMalloc(&tr.blocks[i].ln1_beta, cfg.d_model * sizeof(float));
        cudaMalloc(&tr.blocks[i].ln2_gamma, cfg.d_model * sizeof(float));
        cudaMalloc(&tr.blocks[i].ln2_beta, cfg.d_model * sizeof(float));
    }
    
    DecoderParams  dec = {};
    cudaMalloc(&dec.proj_W, cfg.d_model * PD * sizeof(float));
    cudaMalloc(&dec.proj_b, PD * sizeof(float));

    int step = checkpoint_load(argv[1], &enc, &tr, &dec, &cfg);
    printf("Loaded checkpoint from step %d\n", step);
    fflush(stdout);

    // Load and preprocess input image
    int w, h, c;
    unsigned char* img = stbi_load(argv[2], &w, &h, &c, cfg.channels);
    if (!img) {
        printf("Failed to load image: %s\n", argv[2]);
        return 1;
    }
    printf("Image loaded successfully. w=%d, h=%d, c=%d\n", w, h, c);
    fflush(stdout);
    
    unsigned char* resized_img = (unsigned char*)malloc(cfg.image_size * cfg.image_size * cfg.channels);
    stbir_resize_uint8_linear(img, w, h, 0, resized_img, cfg.image_size, cfg.image_size, 0, (stbir_pixel_layout)cfg.channels);
    
    std::vector<float> h_input(cfg.image_size * cfg.image_size * cfg.channels);
    for (int ch = 0; ch < cfg.channels; ch++) {
        for (int y = 0; y < cfg.image_size; y++) {
            for (int x = 0; x < cfg.image_size; x++) {
                float v = (float)resized_img[(y * cfg.image_size + x) * cfg.channels + ch];
                h_input[ch * cfg.image_size * cfg.image_size + y * cfg.image_size + x] = (v / 127.5f) - 1.0f;
            }
        }
    }
    free(resized_img);

    float *d_input, *d_tokens, *d_tr_out, *d_output;
    int batch_size = 1;
    int img_size = batch_size * cfg.channels * cfg.image_size * cfg.image_size;
    num_patches = (cfg.image_size / cfg.patch_size) * (cfg.image_size / cfg.patch_size);
    int tokens_size = batch_size * num_patches * cfg.d_model;

    cudaMalloc(&d_input,  img_size * sizeof(float));
    cudaMalloc(&d_tokens, tokens_size * sizeof(float));
    cudaMalloc(&d_tr_out, tokens_size * sizeof(float));
    cudaMalloc(&d_output, img_size * sizeof(float));

    cudaMemcpy(d_input, h_input.data(), img_size * sizeof(float), cudaMemcpyHostToDevice);
    printf("Inputs copied to device.\n"); fflush(stdout);

    TransformerCache tr_cache = {};
    tr_cache.ln1_out = new float*[cfg.num_layers];
    tr_cache.attn_out = new float*[cfg.num_layers];
    tr_cache.ln2_out = new float*[cfg.num_layers];
    tr_cache.ffn_mid = new float*[cfg.num_layers];
    tr_cache.ffn_pre_gelu = new float*[cfg.num_layers];
    tr_cache.x_after_attn = new float*[cfg.num_layers];
    tr_cache.attn_caches = new AttentionCache[cfg.num_layers];

    for (int l = 0; l < cfg.num_layers; l++) {
        cudaMalloc(&tr_cache.ln1_out[l], batch_size * num_patches * cfg.d_model * sizeof(float));
        cudaMalloc(&tr_cache.attn_out[l], batch_size * num_patches * cfg.d_model * sizeof(float));
        cudaMalloc(&tr_cache.ln2_out[l], batch_size * num_patches * cfg.d_model * sizeof(float));
        cudaMalloc(&tr_cache.ffn_mid[l], batch_size * num_patches * cfg.d_ff * sizeof(float));
        cudaMalloc(&tr_cache.ffn_pre_gelu[l], batch_size * num_patches * cfg.d_ff * sizeof(float));
        cudaMalloc(&tr_cache.x_after_attn[l], batch_size * num_patches * cfg.d_model * sizeof(float));
        cudaMalloc(&tr_cache.attn_caches[l].Q, batch_size * num_patches * cfg.d_model * sizeof(float));
        cudaMalloc(&tr_cache.attn_caches[l].K, batch_size * num_patches * cfg.d_model * sizeof(float));
        cudaMalloc(&tr_cache.attn_caches[l].V, batch_size * num_patches * cfg.d_model * sizeof(float));
        cudaMalloc(&tr_cache.attn_caches[l].scores, batch_size * cfg.num_heads * num_patches * num_patches * sizeof(float));
        cudaMalloc(&tr_cache.attn_caches[l].input, batch_size * num_patches * cfg.d_model * sizeof(float));
    }

    // Forward pass (no backward needed)
    EncoderConfig e_cfg = {
        cfg.image_size,
        cfg.patch_size,
        cfg.channels,
        cfg.d_model,
        num_patches,
        cfg.patch_size * cfg.patch_size * cfg.channels
    };
    printf("Starting forward pass...\n"); fflush(stdout);
    encoder_forward(d_tokens, d_input, &enc, &e_cfg, 1);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    printf("Encoder done. err=%d\n", err); fflush(stdout);

    // DEBUG: Check d_tokens range
    std::vector<float> h_tokens(tokens_size);
    cudaMemcpy(h_tokens.data(), d_tokens, tokens_size*sizeof(float), cudaMemcpyDeviceToHost);
    float min_tok = 1e9, max_tok = -1e9;
    int nan_count = 0;
    for(int i=0; i<tokens_size; i++) {
        if (isnan(h_tokens[i])) nan_count++;
        if(h_tokens[i] < min_tok) min_tok = h_tokens[i];
        if(h_tokens[i] > max_tok) max_tok = h_tokens[i];
    }
    printf("DEBUG: tokens range [%f, %f], NaNs: %d\n", min_tok, max_tok, nan_count);

    transformer_forward(d_tr_out, &tr_cache, d_tokens, &tr, 1, num_patches);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    printf("Transformer done. err=%d\n", err); fflush(stdout);

    decoder_forward(d_output, d_tr_out, &dec, &e_cfg, 1);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    printf("Decoder done. err=%d\n", err); fflush(stdout);

    // DEBUG: Check d_tr_out range
    std::vector<float> h_tr_out(tokens_size);
    cudaMemcpy(h_tr_out.data(), d_tr_out, tokens_size*sizeof(float), cudaMemcpyDeviceToHost);
    float min_tr = 1e9, max_tr = -1e9;
    for(int i=0; i<tokens_size; i++) {
        if(h_tr_out[i] < min_tr) min_tr = h_tr_out[i];
        if(h_tr_out[i] > max_tr) max_tr = h_tr_out[i];
    }
    printf("DEBUG: tr_out range [%f, %f]\n", min_tr, max_tr);

    // Copy output to CPU and save
    std::vector<float> h_out(img_size);
    cudaMemcpy(h_out.data(), d_output, img_size*sizeof(float), cudaMemcpyDeviceToHost);

    float min_val = 1e9, max_val = -1e9;
    for(int i=0; i<img_size; i++) {
        if(h_out[i] < min_val) min_val = h_out[i];
        if(h_out[i] > max_val) max_val = h_out[i];
    }
    printf("DEBUG: output range [%f, %f]\n", min_val, max_val);

    // Convert [-1,1] float CHW -> [0,255] uint8 HWC
    int H = cfg.image_size, C = cfg.channels;
    std::vector<unsigned char> out_img(H * H * C);
    for (int ch = 0; ch < C; ch++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < H; x++) {
                float v = h_out[ch*H*H + y*H + x];
                out_img[(y*H+x)*C+ch] = (unsigned char)((v+1.0f)*127.5f);
            }

    const char* out_path = (argc > 3) ? argv[3] : "output.png";
    stbi_write_png(out_path, H, H, C, out_img.data(), H * C);
    printf("Saved generated image: %s\n", out_path);

    stbi_image_free(img);
    return 0;
}

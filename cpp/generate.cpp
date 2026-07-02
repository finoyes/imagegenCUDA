#include <stdio.h>
#include "../cuda/utils/checkpoint.h"
#include "../cuda/model/encoder.cuh"
#include "../cuda/model/transformer.cuh"
#include "../cuda/model/decoder.cuh"
#include <vector>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../vendor/stb_image.h"
#include "../vendor/stb_image_write.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: generate <checkpoint.bin> <input_image.jpg> [output.png]\n");
        return 1;
    }

    ModelConfig cfg;
    EncoderParams  enc = {};
    TransformerParams tr = {};
    DecoderParams  dec = {};

    int step = checkpoint_load(argv[1], &enc, &tr, &dec, &cfg);
    printf("Loaded checkpoint from step %d\n", step);

    // Load and preprocess input image
    int w, h, c;
    unsigned char* img = stbi_load(argv[2], &w, &h, &c, cfg.channels);
    // resize, normalize, copy to GPU...

    float *d_input, *d_tokens, *d_tr_out, *d_output;
    int img_size = cfg.batch_size * cfg.channels * cfg.image_size * cfg.image_size;
    cudaMalloc(&d_input,  img_size * sizeof(float));
    cudaMalloc(&d_tokens, /* ... */ 0);
    cudaMalloc(&d_tr_out, /* ... */ 0);
    cudaMalloc(&d_output, img_size * sizeof(float));

    int num_patches = (cfg.image_size / cfg.patch_size) * (cfg.image_size / cfg.patch_size);

    // Forward pass (no backward needed)
    encoder_forward(d_tokens, d_input, &enc, nullptr, 1);
    transformer_forward(d_tr_out, nullptr, d_tokens, &tr, 1, num_patches);
    decoder_forward(d_output, d_tr_out, &dec, nullptr, 1);

    // Copy output to CPU and save
    std::vector<float> h_out(img_size);
    cudaMemcpy(h_out.data(), d_output, img_size*sizeof(float), cudaMemcpyDeviceToHost);

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

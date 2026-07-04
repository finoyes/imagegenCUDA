#include <stdio.h>
#include "cuda/utils/checkpoint.h"
int main(int argc, char** argv) {
    FILE* f = fopen(argv[1], "rb");
    if (!f) return 1;
    uint32_t magic; int version;
    fread(&magic, sizeof(uint32_t), 1, f);
    fread(&version, sizeof(int), 1, f);
    ModelConfig cfg;
    fread(&cfg, sizeof(ModelConfig), 1, f);
    printf("magic=%x, version=%d, cfg.image_size=%d, cfg.patch_size=%d, cfg.channels=%d, cfg.d_model=%d, cfg.num_heads=%d, cfg.num_layers=%d, cfg.d_ff=%d\n", magic, version, cfg.image_size, cfg.patch_size, cfg.channels, cfg.d_model, cfg.num_heads, cfg.num_layers, cfg.d_ff);
    fclose(f);
    return 0;
}

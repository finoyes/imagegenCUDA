#include <stdio.h>
#include <vector>
#include <iostream>
struct ModelConfig {
    int image_size, patch_size, channels;
    int d_model, num_heads, num_layers, d_ff;
    int batch_size;
};
int main() {
    FILE* f = fopen("checkpoints/latest.bin", "rb");
    if(!f) return 1;
    uint32_t magic; int version; ModelConfig cfg; int step;
    fread(&magic, 4, 1, f); fread(&version, 4, 1, f);
    fread(&cfg, sizeof(ModelConfig), 1, f); fread(&step, 4, 1, f);
    printf("magic=%x, version=%d, step=%d\n", magic, version, step);
    int n = 192 * 256;
    std::vector<float> w(n);
    fread(w.data(), sizeof(float), n, f);
    for(int i=0; i<10; i++) printf("%f ", w[i]);
    printf("\n");
    for(int i=0; i<10; i++) printf("%x ", *(unsigned int*)&w[i]);
    printf("\n");
    fclose(f);
    return 0;
}

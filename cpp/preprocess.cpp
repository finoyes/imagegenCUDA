#include "preprocess.h"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../vendor/stb_image.h"
#include "../vendor/stb_image_resize.h"   // stb_image_resize2.h in newer versions
#include <filesystem>
#include <fstream>
#include <stdio.h>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: preprocess <src_dir> <dst_dir>\n");
        return 1;
    }

    PreprocessConfig cfg;
    preprocess_dataset(argv[1], argv[2], cfg);
    return 0;
}

void preprocess_dataset(
    const std::string& src_dir,
    const std::string& dst_dir,
    const PreprocessConfig& cfg)
{
    fs::create_directories(dst_dir);
    int S = cfg.target_size, C = cfg.channels;

    std::vector<unsigned char> resized(S * S * C);
    std::vector<float> normalized(S * S * C);

    int count = 0;
    for (auto& entry : fs::directory_iterator(src_dir)) {
        std::string ext = entry.path().extension().string();
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png") continue;

        int w, h, c;
        unsigned char* data = stbi_load(entry.path().string().c_str(), &w, &h, &c, C);
        if (!data) { fprintf(stderr, "skip %s\n", entry.path().string().c_str()); continue; }

        // Resize using stb_image_resize
        stbir_resize_uint8_linear(data, w, h, 0, resized.data(), S, S, 0, (stbir_pixel_layout)C);
        stbi_image_free(data);

        // Convert to float, channels-first (C x H x W), normalize to [-1, 1]
        for (int ch = 0; ch < C; ch++)
            for (int y = 0; y < S; y++)
                for (int x = 0; x < S; x++) {
                    unsigned char px = resized[(y * S + x) * C + ch];
                    float val = cfg.normalize ? (px / 127.5f - 1.0f) : (px / 255.0f);
                    normalized[ch * S * S + y * S + x] = val;
                }

        // Write to .bin file — just raw floats, no header
        std::string out_path = dst_dir + "/" + entry.path().stem().string() + ".bin";
        std::ofstream out(out_path, std::ios::binary);
        out.write((char*)normalized.data(), normalized.size() * sizeof(float));
        count++;

        if (count % 100 == 0)
            printf("Processed %d images...\n", count);
    }
    printf("Done. Processed %d images -> %s\n", count, dst_dir.c_str());
}

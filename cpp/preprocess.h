#pragma once
#include <string>

struct PreprocessConfig {
    int target_size  = 64;     // resize images to target_size x target_size
    int channels     = 3;
    bool normalize   = true;   // map pixels to [-1, 1]
};

// Process all images in src_dir, write .bin files to dst_dir
void preprocess_dataset(
    const std::string& src_dir,
    const std::string& dst_dir,
    const PreprocessConfig& cfg
);

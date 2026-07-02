#pragma once
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cuda_runtime.h>

struct AugConfig {
    bool   random_flip   = true;
    float  crop_scale    = 0.9f;    // crop to 90% then resize
    float  mean[3]       = {0.5f, 0.5f, 0.5f};
    float  std[3]        = {0.5f, 0.5f, 0.5f};
};

class DataLoader {
public:
    DataLoader(const std::string& data_dir,
               int batch_size, int image_size, int channels,
               const AugConfig& aug = {});
    ~DataLoader();

    // Returns a device pointer to the next batch: (batch x C x H x W)
    // Blocks until the batch is ready
    float* next_batch();

    int num_batches() const;
    void reset();    // reshuffle and restart epoch

private:
    void prefetch_worker();
    void load_and_augment(const std::string& path, float* out);
    void augment(float* img);

    std::string  data_dir_;
    int          batch_size_, image_size_, channels_;
    AugConfig    aug_;

    std::vector<std::string> file_list_;
    std::vector<int>         indices_;
    int                      current_idx_;

    // Double-buffering: one buffer being consumed, one being filled
    float* pinned_buf_[2];   // page-locked CPU buffers
    float* gpu_buf_[2];      // corresponding GPU buffers
    int    active_buf_;

    cudaStream_t transfer_stream_;

    std::thread             worker_;
    std::mutex              mutex_;
    std::condition_variable cv_ready_, cv_consumed_;
    bool                    next_ready_ = false;
    bool                    stop_       = false;
};
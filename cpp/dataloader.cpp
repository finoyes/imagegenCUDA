#include "dataloader.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../vendor/stb_image.h"
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <random>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace fs = std::filesystem;

DataLoader::DataLoader(const std::string& dir,
                       int batch_size, int image_size, int channels,
                       const AugConfig& aug)
    : data_dir_(dir), batch_size_(batch_size),
      image_size_(image_size), channels_(channels), aug_(aug),
      current_idx_(0), active_buf_(0)
{
    // Gather supported dataset files.
    for (auto& entry : fs::directory_iterator(dir))
        if (entry.path().extension() == ".jpg" ||
            entry.path().extension() == ".jpeg" ||
            entry.path().extension() == ".png")
            file_list_.push_back(entry.path().string());
        else if (entry.path().extension() == ".bin")
            file_list_.push_back(entry.path().string());

    if (file_list_.empty()) {
        throw std::runtime_error("DataLoader found no supported files in " + dir +
                                 " (expected .bin, .jpg, .jpeg, or .png)");
    }

    indices_.resize(file_list_.size());
    std::iota(indices_.begin(), indices_.end(), 0);

    size_t batch_bytes = (size_t)batch_size * channels * image_size * image_size
                         * sizeof(float);

    // Allocate pinned (page-locked) CPU buffers — faster DMA to GPU
    cudaMallocHost(&pinned_buf_[0], batch_bytes);
    cudaMallocHost(&pinned_buf_[1], batch_bytes);
    cudaMalloc(&gpu_buf_[0], batch_bytes);
    cudaMalloc(&gpu_buf_[1], batch_bytes);

    cudaStreamCreate(&transfer_stream_);

    // Start prefetch thread
    worker_ = std::thread(&DataLoader::prefetch_worker, this);
}

DataLoader::~DataLoader() {
    { std::lock_guard<std::mutex> lk(mutex_); stop_ = true; }
    cv_consumed_.notify_all();
    worker_.join();
    cudaFreeHost(pinned_buf_[0]); cudaFreeHost(pinned_buf_[1]);
    cudaFree(gpu_buf_[0]);        cudaFree(gpu_buf_[1]);
    cudaStreamDestroy(transfer_stream_);
}

void DataLoader::load_and_augment(const std::string& path, float* out) {
    if (fs::path(path).extension() == ".bin") {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) {
            throw std::runtime_error("Failed to open dataset file: " + path);
        }

        const size_t expected_floats = (size_t)channels_ * image_size_ * image_size_;
        in.read(reinterpret_cast<char*>(out), expected_floats * sizeof(float));
        if (in.gcount() != static_cast<std::streamsize>(expected_floats * sizeof(float))) {
            fprintf(stderr, "[DataLoader] Bad .bin file (expected %zu bytes, got %lld): %s\n",
                    expected_floats * sizeof(float),
                    (long long)in.gcount(), path.c_str());
            throw std::runtime_error("Unexpected size while reading dataset file: " + path);
        }
    } else {
        int w, h, c;
        unsigned char* data = stbi_load(path.c_str(), &w, &h, &c, channels_);
        if (!data) return;

        // Resize to image_size_ x image_size_ using nearest-neighbor (simple)
        // In production: use bilinear interpolation
        for (int ch = 0; ch < channels_; ch++)
            for (int y = 0; y < image_size_; y++)
                for (int x = 0; x < image_size_; x++) {
                    int src_y = y * h / image_size_;
                    int src_x = x * w / image_size_;
                    int src_idx = (src_y * w + src_x) * channels_ + ch;
                    // Normalize to [-1, 1]
                    out[ch * image_size_ * image_size_ + y * image_size_ + x]
                        = (data[src_idx] / 255.0f - aug_.mean[ch]) / aug_.std[ch];
                }

        stbi_image_free(data);
    }
    augment(out);   // apply random augmentations
}

void DataLoader::augment(float* img) {
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // Random horizontal flip
    if (aug_.random_flip && dist(rng) > 0.5f) {
        int H = image_size_, W = image_size_;
        for (int c = 0; c < channels_; c++)
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W / 2; x++) {
                    float tmp = img[c*H*W + y*W + x];
                    img[c*H*W + y*W + x]       = img[c*H*W + y*W + (W-1-x)];
                    img[c*H*W + y*W + (W-1-x)] = tmp;
                }
    }
}

void DataLoader::prefetch_worker() {
    int fill_buf = 1;   // fill the non-active buffer
    while (true) {
        {
            std::unique_lock<std::mutex> lk(mutex_);
            cv_consumed_.wait(lk, [&]{ return !next_ready_ || stop_; });
            if (stop_) return;
        }

        // Load a batch into the fill buffer
        int bsz = batch_size_;
        size_t img_floats = (size_t)channels_ * image_size_ * image_size_;
        for (int i = 0; i < bsz; i++) {
            // reset() was previously called here without the mutex, which is a
            // data race. Instead we inline the reset while no other thread is
            // touching these fields (the main thread is waiting on cv_ready_).
            if (current_idx_ >= (int)indices_.size()) {
                std::shuffle(indices_.begin(), indices_.end(),
                             std::mt19937{std::random_device{}()});
                current_idx_ = 0;
            }
            std::string path = file_list_[indices_[current_idx_++]];
            load_and_augment(path, pinned_buf_[fill_buf] + i * img_floats);
        }

        // Async copy pinned CPU → GPU
        size_t bytes = (size_t)bsz * img_floats * sizeof(float);
        cudaMemcpyAsync(gpu_buf_[fill_buf], pinned_buf_[fill_buf],
                        bytes, cudaMemcpyHostToDevice, transfer_stream_);
        cudaStreamSynchronize(transfer_stream_);

        { std::lock_guard<std::mutex> lk(mutex_); next_ready_ = true; fill_buf ^= 1; }
        cv_ready_.notify_one();
    }
}

float* DataLoader::next_batch() {
    std::unique_lock<std::mutex> lk(mutex_);
    cv_ready_.wait(lk, [&]{ return next_ready_; });
    next_ready_ = false;
    active_buf_ ^= 1;
    cv_consumed_.notify_one();
    return gpu_buf_[active_buf_];
}

void DataLoader::reset() {
    // NOTE: Only call this from within prefetch_worker(), where indices_ and
    // current_idx_ are already accessed under the protocol of the condition
    // variables (main thread is blocked waiting on cv_ready_).
    std::shuffle(indices_.begin(), indices_.end(),
                 std::mt19937{std::random_device{}()});
    current_idx_ = 0;
}

int DataLoader::num_batches() const {
    return (int)file_list_.size() / batch_size_;
}

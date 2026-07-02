#pragma once
#include <string>
#include <fstream>
#include <cuda_runtime.h>

class Logger {
public:
    Logger(const std::string& log_dir, int save_every_n_steps = 500);
    ~Logger();

    void log_loss(int step, int epoch, float loss, float lr);

    // Save a batch of reconstructed images to PNG
    // pred: (batch x C x H x W) device pointer, values in [-1, 1]
    void save_samples(int step, const float* d_pred, const float* d_target,
                      int batch, int channels, int image_size);

private:
    std::string  log_dir_;
    int          save_every_;
    std::ofstream loss_csv_;
};

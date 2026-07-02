#include "logger.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../vendor/stb_image_write.h"
#include <filesystem>
#include <vector>
#include <stdio.h>

namespace fs = std::filesystem;

Logger::Logger(const std::string& log_dir, int save_every)
    : log_dir_(log_dir), save_every_(save_every)
{
    fs::create_directories(log_dir);
    fs::create_directories(log_dir + "/samples");
    loss_csv_.open(log_dir + "/loss.csv");
    loss_csv_ << "step,epoch,loss,lr\n";
}

Logger::~Logger() { loss_csv_.close(); }

void Logger::log_loss(int step, int epoch, float loss, float lr) {
    loss_csv_ << step << "," << epoch << "," << loss << "," << lr << "\n";
    loss_csv_.flush();
    printf("[step %6d | epoch %3d] loss=%.6f lr=%.2e\n", step, epoch, loss, lr);
}

void Logger::save_samples(
    int step, const float* d_pred, const float* d_target,
    int batch, int C, int H)
{
    if (step % save_every_ != 0) return;

    int n = batch * C * H * H;
    std::vector<float> h_pred(n), h_target(n);
    cudaMemcpy(h_pred.data(),   d_pred,   n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_target.data(), d_target, n*sizeof(float), cudaMemcpyDeviceToHost);

    // Save first 4 images from the batch (predicted + target side by side)
    int save_n = std::min(batch, 4);
    for (int b = 0; b < save_n; b++) {
        // Convert CHW float [-1,1] -> HWC uint8 [0,255]
        std::vector<unsigned char> pred_u8(H * H * C);
        std::vector<unsigned char> tgt_u8(H * H * C);
        for (int ch = 0; ch < C; ch++)
            for (int y = 0; y < H; y++)
                for (int x = 0; x < H; x++) {
                    int ci = ch*H*H + y*H + x;
                    int hw = (y*H + x)*C + ch;
                    pred_u8[hw] = (unsigned char)((h_pred  [b*C*H*H + ci] + 1.0f) * 127.5f);
                    tgt_u8 [hw] = (unsigned char)((h_target[b*C*H*H + ci] + 1.0f) * 127.5f);
                }

        char path[256];
        snprintf(path, 256, "%s/samples/step%06d_b%d_pred.png",
                 log_dir_.c_str(), step, b);
        stbi_write_png(path, H, H, C, pred_u8.data(), H * C);

        snprintf(path, 256, "%s/samples/step%06d_b%d_target.png",
                 log_dir_.c_str(), step, b);
        stbi_write_png(path, H, H, C, tgt_u8.data(), H * C);
    }
    printf("Saved sample images at step %d\n", step);
}

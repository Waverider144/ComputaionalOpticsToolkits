#pragma once
#include <torch/torch.h>
#include "WavePropagator.hpp"

class ASM : public WavePropagator {
public:
    // 只有这一个构造函数
    ASM(int N, double wl, double dx, double z, torch::Device device);

    // 只有这一个覆盖声明
    torch::Tensor propagate(torch::Tensor input, bool is_phase) override;

private:
    int N;
    double wl, dx, z;
    torch::Device device; 
    torch::Tensor H;

    void precompute_transfer_function();
    torch::Tensor fftshift(torch::Tensor x);
};
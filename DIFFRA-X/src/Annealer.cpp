#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include "Annealer.hpp"
#include "Config.hpp"

Annealer::Annealer(int N) : N(N) {}

torch::Tensor Annealer::apply_gaussian_blur(const torch::Tensor& target, double sigma) {
    if (sigma <= 0.05) return target.clone();

    torch::NoGradGuard no_grad;

    // 1. 根据 sigma 确定核大小 (通常 6*sigma + 1)
    int k_size = static_cast<int>(std::ceil(sigma * 3.0)) * 2 + 1;
    if (k_size < 3) k_size = 3;

    // 2. 创建一维高斯向量
    auto coords = torch::arange(k_size, torch::kFloat32) - (k_size - 1) / 2.0;
    auto g = torch::exp(-(coords.pow(2)) / (2 * sigma * sigma));
    g = g / g.sum();

    // 3. 利用外积构造二维高斯核 [1, 1, K, K]
    auto kernel2d = torch::outer(g, g).view({1, 1, k_size, k_size}).to(target.device());

    // 4. 执行卷积 (使用 replication_pad 保持边缘稳定)
    int pad = k_size / 2;
    auto input = target.view({1, 1, N, N});
    auto padded = torch::replication_pad2d(input, {pad, pad, pad, pad});
    auto blurred = torch::conv2d(padded, kernel2d);

    return blurred.view({N, N});
}


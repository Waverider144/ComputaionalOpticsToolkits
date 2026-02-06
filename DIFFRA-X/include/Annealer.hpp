#pragma once
#include <torch/torch.h>
#include <string>

class Annealer {
public:
    Annealer(int N);

    /**
     * 对目标图像进行高斯平滑
     * @param target 原始二值化目标图
     * @param sigma 高斯核标准差
     */
    torch::Tensor apply_gaussian_blur(const torch::Tensor& target, double sigma);


private:
    int N;
};
#pragma once
#include <torch/torch.h>
#include "ASM.hpp"
#include "Annealer.hpp"

class DiffEngine {
public:
    // 构造函数接收物理引擎的引用
    DiffEngine(ASM& p_asm, Annealer& p_annealer);

    /**
     * 执行一次完整的梯度计算流程
     * @param mask_raw 待优化的原始参数（实数张量）
     * @param target_orig 原始文字位图目标
     * @param sigma 当前 PID 控制的高斯模糊系数
     * @param beta 当前二值化惩罚权重
     * @return 包含 Loss 信息和计算出的梯度
     */

    struct DiffResult {
    torch::Tensor recon_intensity; // 物理传播后的结果
    torch::Tensor mse_tensor;      // 基础 MSE (带梯度的 Tensor)
    float mse_loss;                // 打印用的数值
    };

    DiffResult compute_gradients(torch::Tensor& mask_raw, 
                                    const torch::Tensor& target_orig, 
                                    double sigma, 
                                    double beta);

private:
    ASM& phy;
    Annealer& annealer;
};
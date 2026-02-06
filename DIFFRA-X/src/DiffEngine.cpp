#include "DiffEngine.hpp"

DiffEngine::DiffEngine(ASM& p_asm, Annealer& p_annealer) 
    : phy(p_asm), annealer(p_annealer) {}

DiffEngine::DiffResult DiffEngine::compute_gradients(
    torch::Tensor& mask_raw, 
    const torch::Tensor& target_orig, 
    double sigma, 
    double beta) 
{
    // 1. 确保参数有梯度记录权限
    if (!mask_raw.requires_grad()) {
        mask_raw.set_requires_grad(true);
    }
    if (mask_raw.grad().defined()) {
        mask_raw.grad().zero_();
    }

    // 2. 策略层：通过退火算法获取当前平滑后的目标
    // 注意：目标图像不需要梯度
    torch::Tensor target_smoothed;
    {
        torch::NoGradGuard no_grad; 
        target_smoothed = annealer.apply_gaussian_blur(target_orig, sigma);
    }

    // 3. 正向传播：从参数映射到物理场
    // 我们这里假设是相位型衍射屏，使用 Sigmoid 映射到 [0, 1] 再乘以 2*PI
    auto mask = torch::sigmoid(mask_raw);
    auto phase = mask * 2.0 * M_PI;

    // 调用 ASM 物理层
    auto output_field = phy.propagate(phase, true); // true 代表相位输入
    auto intensity = output_field.abs().pow(2);
    
    // 归一化光强，确保 Loss 计算的尺度稳定
    intensity = intensity / (intensity.max() + 1e-8);

    // 4. 计算损失函数 (Loss Function)
    auto mse = torch::mse_loss(intensity, target_smoothed);
    
    // 二值化惩罚项：mask * (1 - mask) 在 0 和 1 处最小
    auto penalty = torch::mean(mask * (1.0 - mask));
    
    auto total_loss = mse + beta * penalty;

    // 5. 反向传播 (核心自动微分步骤)
    // 这一步等效于伴随状态法中的：计算伴随源 -> 反向传播伴随场 -> 提取重叠积分梯度
    
    auto mse_tensor = torch::mse_loss(intensity, target_smoothed);

    DiffEngine::DiffResult res; 
    res.recon_intensity = intensity;
    res.mse_tensor = mse_tensor;
    res.mse_loss = mse_tensor.item<float>();
    return res;
}
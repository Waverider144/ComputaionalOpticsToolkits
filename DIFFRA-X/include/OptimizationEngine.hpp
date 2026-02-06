#pragma once
#include <torch/torch.h>
#include <functional>

// [START REFACTOR: OPTIMIZATION_STRATEGY]
class OptimizationEngine {
public:
    virtual ~OptimizationEngine() = default;
    // loss_func 返回需要 backward 的 Tensor
    virtual void optimize_step(torch::Tensor& mask, 
                              torch::optim::Optimizer& opt,
                              std::function<torch::Tensor()> loss_func) = 0;
};

// 默认 Adjoint 方法 (利用 LibTorch 自动微分)
class AdjointEngine : public OptimizationEngine {
public:
    void optimize_step(torch::Tensor& mask, torch::optim::Optimizer& opt, std::function<torch::Tensor()> loss_func) override {
        opt.zero_grad();
        auto loss = loss_func(); // 执行前向计算
        loss.backward();         // 反向传播
        opt.step();
    }
};

// 数值微分引擎 (用于 Benchmark)
class NumericDiffEngine : public OptimizationEngine {
    // 预留位置：你可以在这里通过扰动 mask[i][j] 来手动计算梯度
};
// [END REFACTOR: OPTIMIZATION_STRATEGY]
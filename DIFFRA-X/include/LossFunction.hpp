#pragma once
#include <torch/torch.h>
#include <memory>
#include <external/json.hpp> 
#include <deque>
#include <numeric>

#include "DiffEngine.hpp"
#include "Config.hpp"
#include "OptimizationEngine.hpp"

class PhaseOptimizer {
public:
    struct StepResult {
        float loss;
        double sigma;
        double beta;
        bool should_stop;
    };

    PhaseOptimizer(std::shared_ptr<DiffEngine> engine, torch::Tensor target, const Config& config)
        : engine_(engine), target_(target) {

        // 保存一份 raw 引用以便后续逻辑使用，或者直接使用 config.raw
        auto& raw = config.raw;
        auto& opt_json = raw["optimizer"];

        // 1. 动态选择 Sigma 控制器 (修正：使用 config.raw 访问 json 接口)

        std::string mode = opt_json.value("sigma_mode", "annealing");

        if (mode == "pid") {
            // 使用 j 里的 pid 配置
            sigma_ctrl = std::make_unique<PIDController>(raw.value("pid", nlohmann::json::object()));
        } 
        else {
            // 使用 j 里的 annealing 配置Config
            sigma_ctrl = std::make_unique<AnnealingController>(raw.value("annealing", nlohmann::json::object()));
        }

        // 2. 动态选择优化引擎
        if (opt_json.value("engine_mode", "adjoint") == "numeric") {
            // opt_engine = std::make_unique<NumericDiffEngine>();
        } else {
            opt_engine = std::make_unique<AdjointEngine>();
        }

        // 修正：从类实例的 raw json 获取值
        stop_threshold = opt_json.value("stop_threshold", 1e-6);
        window_size = opt_json.value("window_size", 20);

        // 为了 schedule_beta 能用，保存一份必要的成员副本
        this->max_steps_cache = opt_json.value("max_steps", 1000);
        this->final_beta_cache = opt_json.value("final_beta", 1.0);
    }

    StepResult step(torch::Tensor& mask_raw, int current_step, torch::optim::Optimizer& opt) {
        opt.zero_grad();
        double sigma = sigma_ctrl->update(last_loss, current_step);
        double beta = schedule_beta(current_step); 

        opt_engine->optimize_step(mask_raw, opt, [&]() {
            auto res = engine_->compute_gradients(mask_raw, target_, sigma, beta);
            last_loss = res.mse_loss;
            last_res_tensor = res.mse_tensor;
            return res.mse_tensor;
        });

        update_history(last_loss);
        return {last_loss, sigma, beta, check_convergence()};
    }

private:
    std::shared_ptr<DiffEngine> engine_;
    torch::Tensor target_;
    torch::Tensor last_res_tensor;
    
    std::unique_ptr<Controller> sigma_ctrl;
    std::unique_ptr<OptimizationEngine> opt_engine;

    float last_loss = 1.0f;
    double stop_threshold;  
    size_t window_size;     
    std::deque<float> loss_history;

    // 缓存变量，解决 Config:: 静态成员报错
    int max_steps_cache;
    double final_beta_cache;

    double schedule_beta(int step) {
        // 修正：改用成员变量而非静态 Config:: 调用
        if (step > max_steps_cache * 0.4) 
            return std::min(final_beta_cache, (step - max_steps_cache * 0.4) * 0.002);
        return 0.0;
    }

    void update_history(float loss) {
        loss_history.push_back(loss);
        if (loss_history.size() > window_size) loss_history.pop_front();
    }

    bool check_convergence() {
        // if (loss_history.size() < window_size) return false;
        // float sum = std::accumulate(loss_history.begin(), loss_history.end(), 0.0f);
        // float mean = sum / (float)window_size;
        // float sq_sum = std::inner_product(loss_history.begin(), loss_history.end(), loss_history.begin(), 0.0f);
        // float variance = std::max(0.0f, sq_sum / (float)window_size - mean * mean);
        // float stdev = std::sqrt(variance);
        // return (stdev < stop_threshold);
        return false;
    }
};
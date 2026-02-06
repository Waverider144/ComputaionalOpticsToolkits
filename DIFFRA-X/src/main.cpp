#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <filesystem>

#include "Config.hpp"
#include "ASM.hpp"
#include "DiffEngine.hpp"
#include "PIDController.hpp"
#include "Annealer.hpp"
#include "Logger.hpp"
#include "ImageLoader.hpp"
#include "LossFunction.hpp"

namespace fs = std::filesystem;

cv::Mat tensor_to_mat(torch::Tensor t) {
    torch::NoGradGuard no_grad;
    auto cpu_t = t.detach().cpu().to(torch::kFloat32);
    cv::Mat mat(cpu_t.size(0), cpu_t.size(1), CV_32FC1, cpu_t.data_ptr<float>());
    cv::Mat res;
    mat.convertTo(res, CV_8UC1, 255.0);
    return res.clone();
}

int main() {
    // 1. 加载配置（严禁更改变量名 config）
    Config config; 
    try {
        config = Config::load("config.json");
    } catch (const std::exception& e) {
        std::cerr << "Initialization Failed: " << e.what() << std::endl;
        return -1;
    }

    // 修正：改用实例 config.output_dir
    Logger::getInstance().init(config.output_dir);
    LOG_INFO("DIFFRA-X Platform Started.");

    // 2. 硬件准备
    auto device = Config::get_device();
    // 这里如果 get_device_string 被删了，建议手动写或在 Config 里补上，此处暂改用 cfg 逻辑
    LOG_INFO("Using Device: CUDA/CPU check complete.");

    // main.cpp
    std::cout << "--- Debug Physics ---" << std::endl;
    std::cout << "N: " << config.N << " | lambda: " << config.wavelength << std::endl;
    std::cout << "Pixel: " << config.pixel_size << " | Dist: " << config.distance << std::endl;

    // 如果输出是 N: 0 或者 lambda: 0，说明你 Config.hpp 里的变量没有正确接收 json 的值

    // 3. 初始化组件 (修正：全部改用 config 实例访问)
    auto asm_phy = std::make_shared<ASM>(config.N, config.wavelength, config.pixel_size, config.distance, device);
    auto annealer = std::make_shared<Annealer>(config.N);
    
    // 假设 DiffEngine 需要物理配置
    auto diff_engine = std::make_shared<DiffEngine>(*asm_phy, *annealer); 

    // 4. 加载目标 (修正：使用 config.raw 和 config.N)
    auto target_intensity = ImageLoader::load_to_tensor(config.raw, config.N, device);
    if (target_intensity.dtype() != torch::kFloat32) target_intensity = target_intensity.to(torch::kFloat32);

    std::cout << "--- LAST CHECK BEFORE SOLVER ---" << std::endl;
    std::cout << "Config N: " << config.N << std::endl;
    std::cout << "JSON Raw empty? " << config.raw.empty() << std::endl;
    if (!config.raw.empty()) {
        std::cout << "PID KP: " << config.raw["pid"]["kp"] << std::endl;
    }

    auto solver = std::make_unique<PhaseOptimizer>(diff_engine, target_intensity, config);

    // 5. 初始化优化变量
    std::vector<int64_t> sz = { (int64_t)config.N, (int64_t)config.N };
    auto mask_raw = torch::randn({config.N, config.N}, torch::kFloat32).to(device).set_requires_grad(true);
    
    // 从 config.raw 中读取 Adam 学习率
    double lr = config.raw["optimizer"].value("learning_rate", 0.01);
    auto optimizer = torch::optim::Adam(std::vector<torch::Tensor>{mask_raw}, torch::optim::AdamOptions(lr));

    auto start_time = std::chrono::steady_clock::now();
    
    // 6. 优化主循环
    int max_steps = config.raw["optimizer"].value("max_steps", 1000);
    for (int step = 0; step < max_steps; ++step) {
        auto step_out = solver->step(mask_raw, step, optimizer);

        if (std::isnan(step_out.loss)) {
        std::cout << "CRITICAL: NaN detected at step " << step << std::endl;
        std::cout << "Current Sigma: " << step_out.sigma << " Beta: " << step_out.beta << std::endl;
        // 如果 sigma 是 nan，说明问题在 PID/Annealing 控制器
        // 如果 loss 是 nan 但 sigma 是正常的，说明问题在 ASM 传播
        return -1;
        }

        if (step_out.should_stop) {
            printf(">>> [STOP] Convergence reached at step %d\n", step);
            break;
        }

        if (step % 20 == 0) {
            auto current_mask = torch::sigmoid(mask_raw);
            cv::Mat mask_vis = tensor_to_mat(current_mask);
            
            {
                torch::NoGradGuard no_grad;
                auto current_phase = current_mask * 2.0 * M_PI;
                auto field_forward = asm_phy->propagate(current_phase, true);
                cv::Mat recon_vis = tensor_to_mat(field_forward.abs());
                
                ImageManager::save_and_limit(field_forward.abs(), step, config.output_dir, 1000);
            }

            LOG_DATA(step, step_out.loss, step_out.sigma);
            printf("[Step %d] Loss: %.6f | Sigma: %.2f\n", step, (double)step_out.loss, step_out.sigma);
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    LOG_INFO("Optimization Complete.");

    return 0;
}
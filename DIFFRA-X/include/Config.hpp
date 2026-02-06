#pragma once
#include <string>
#include <torch/torch.h>
#include "external/json.hpp"
#include <fstream>
#include <filesystem>

using json = nlohmann::json;

class Config {
public:
    // --- 成员变量 (每个只定义一次，且没有 static) ---
    int N=512;
    double wavelength=532e-9;
    double pixel_size=8e-6;
    double distance=0.05;
    int max_steps=0;
    double learning_rate=0;
    double final_beta=0;
    std::string output_dir="outputs";
    json raw; // 存储原始 JSON 数据供各组件使用

    // --- 静态工具函数 (可以在 main 直接通过 Config:: 调用) ---
    static torch::Device get_device() {
        return torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    }

    static std::string get_device_string() {
        return torch::cuda::is_available() ? "NVIDIA GPU (CUDA)" : "CPU";
    }

    // --- 加载函数 ---
    static Config load(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) throw std::runtime_error("Config file not found: " + path);
        
        json data = json::parse(f);
        Config cfg;
        cfg.raw = data;

        // 物理参数映射
        auto& phys = data["physics"];
        cfg.N = phys.value("N", 512);
        cfg.wavelength = phys.value("wavelength", 532e-9);
        cfg.pixel_size = phys.value("pixel_size", 8e-6);
        cfg.distance = phys.value("distance", 0.05);

        // 优化参数映射
        auto& opt = data["optimizer"];
        cfg.max_steps = opt.value("max_steps", 1000);
        cfg.learning_rate = opt.value("learning_rate", 0.01);
        cfg.final_beta = opt.value("final_beta", 1.0);

        // 路径映射
        cfg.output_dir = data["paths"].value("output_dir", "outputs");
        std::filesystem::create_directories(cfg.output_dir);

        return cfg;
    }
};
#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <mutex>
#include <filesystem>
#include <algorithm>

enum class LogLevel { INFO, WARNING, ERROR, DATA };

class ImageManager {
public:
    static void save_and_limit(torch::Tensor field_abs, int step, const std::string& dir, size_t max_files = 1000) {
        namespace fs = std::filesystem;
        if (!fs::exists(dir)) fs::create_directories(dir);

        // 1. Tensor 转 Mat (封装之前的转换逻辑)
        torch::NoGradGuard no_grad;
        auto cpu_t = field_abs.detach().cpu().to(torch::kFloat32);
        cv::Mat mat(cpu_t.size(0), cpu_t.size(1), CV_32FC1, cpu_t.data_ptr<float>());
        cv::Mat recon_vis;
        mat.convertTo(recon_vis, CV_8UC1, 255.0);

        // 2. 构造路径并保存
        std::string full_path = dir + "/recon_step_" + std::to_string(step) + ".png";
        cv::imwrite(full_path, recon_vis);

        // 3. 自动清理旧文件
        std::vector<fs::directory_entry> png_files;
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".png") {
                png_files.push_back(entry);
            }
        }

        if (png_files.size() > max_files) {
            // 按修改时间升序排序（旧的在前）
            std::sort(png_files.begin(), png_files.end(), [](const auto& a, const auto& b) {
                return a.last_write_time() < b.last_write_time();
            });

            // 删除多余的旧文件
            for (size_t i = 0; i < png_files.size() - max_files; ++i) {
                fs::remove(png_files[i].path());
            }
        }
    }
};

class Logger {
public:
    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }
    void init(const std::string& folder) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::filesystem::create_directories(folder);
        log_file_.open(folder + "/diffra_x.log", std::ios::app);
        data_file_.open(folder + "/metrics.csv", std::ios::app);
    }
    void log(LogLevel level, const std::string& message) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::string label;
        switch (level) {
            case LogLevel::INFO:    label = "[INFO] "; break;
            case LogLevel::WARNING: label = "[WARN] "; break;
            case LogLevel::ERROR:   label = "[ERR ] "; break;
            case LogLevel::DATA:    label = "[DATA] "; break;
        }
        std::cout << label << message << std::endl;
        if (log_file_.is_open()) log_file_ << label << message << std::endl;
    }
    void log_metrics(int step, double loss, double sigma) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (data_file_.is_open()) data_file_ << step << "," << loss << "," << sigma << "\n";
    }

private:
    Logger() = default;
    std::ofstream log_file_;
    std::ofstream data_file_;
    std::mutex mutex_;
};

#define LOG_INFO(msg) Logger::getInstance().log(LogLevel::INFO, msg)
#define LOG_DATA(s, l, sig) Logger::getInstance().log_metrics(s, l, sig)
#define LOG_ERROR(msg) Logger::getInstance().log(LogLevel::ERROR, msg)
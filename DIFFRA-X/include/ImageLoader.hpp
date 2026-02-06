#pragma once
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <string>
#include "external/json.hpp" // 必须包含

class ImageLoader {
public:
    /**
     * @brief 加载图像并进行标准化处理
     * @param path 图像路径
     * @param N 目标网格尺寸 (N x N)
     * @param device 计算设备 (CPU/CUDA)
     * @param keep_aspect 是否保持纵横比（如果为 true，则进行 Letterboxing 填充）
     */
    static torch::Tensor load_to_tensor(const std::string& path, int N, torch::Device device, bool keep_aspect = true);
    static torch::Tensor load_to_tensor(const nlohmann::json& config, int N, torch::Device device);

private:
    // 内部辅助函数：处理“黑边填充”逻辑
    static cv::Mat render_text(const std::string& text, int N, double font_scale, int thickness);
    static cv::Mat letterbox(const cv::Mat& src, int N);
};

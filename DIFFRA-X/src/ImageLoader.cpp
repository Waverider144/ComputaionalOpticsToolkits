#include "ImageLoader.hpp"
#include "Logger.hpp" // 假设你已经有了之前写的 Logger
#include <iostream>

// 内部私有函数：实现居中填充（Letterboxing）
cv::Mat ImageLoader::letterbox(const cv::Mat& src, int N) {
    int w = src.cols;
    int h = src.rows;
    double scale = std::min((double)N / w, (double)N / h);
    
    int nw = (int)(w * scale);
    int nh = (int)(h * scale);

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(nw, nh), 0, 0, cv::INTER_AREA);

    // 创建黑色背景画布
    cv::Mat canvas = cv::Mat::zeros(N, N, CV_8UC1);
    
    // 计算居中坐标
    int dx = (N - nw) / 2;
    int dy = (N - nh) / 2;

    resized.copyTo(canvas(cv::Rect(dx, dy, nw, nh)));
    return canvas;
}

// 内部私有函数：将文字渲染成位图
cv::Mat ImageLoader::render_text(const std::string& text, int N, double font_scale, int thickness) {
    cv::Mat canvas = cv::Mat::zeros(N, N, CV_8UC1);
    
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
    
    // 计算文字位置使其居中 (OpenCV 坐标系左上角为原点)
    cv::Point text_org((N - text_size.width) / 2, (N + text_size.height) / 2);
    
    cv::putText(canvas, text, text_org, cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255), thickness);
    return canvas;
}

// 核心公开接口
torch::Tensor ImageLoader::load_to_tensor(const nlohmann::json& config, int N, torch::Device device) {
    std::string mode = config["input"]["mode"];
    cv::Mat processed;

    if (mode == "text") {
        std::string content = config["input"].value("text_content", "HARVEY");
        double scale = config["input"].value("font_scale", 1.5);
        int thick = config["input"].value("thickness", 2);
        
        processed = render_text(content, N, scale, thick);
        LOG_INFO("Target initialized from TEXT: " + content);
    } 
    else if (mode == "image") {
        std::string path = config["input"]["image_path"];
        cv::Mat raw = cv::imread(path, cv::IMREAD_GRAYSCALE);
        
        if (raw.empty()) {
            LOG_ERROR("Failed to load image: " + path);
            // 如果加载失败，给个默认报错图
            processed = render_text("IMG ERROR", N, 1.0, 2);
        } else {
            processed = letterbox(raw, N);
            LOG_INFO("Target initialized from IMAGE: " + path);
        }
    }

    // 可选：如果 JSON 里设置了 invert，则反转颜色
    if (config["input"].value("invert", false)) {
        processed = 255 - processed;
        LOG_INFO("Target color inverted.");
    }

    // 归一化并转换为 Torch Tensor
    processed.convertTo(processed, CV_32FC1, 1.0 / 255.0);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor t = torch::from_blob(processed.data, {N, N}, options);
    
    // clone 拷贝到计算设备 (CUDA/CPU) 并断开与 OpenCV 内存的联系
    return t.to(device).clone();
}
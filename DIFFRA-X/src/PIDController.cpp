#include "PIDController.hpp"

PIDController::PIDController(const nlohmann::json& c) {
    kp_ = c.value("kp", 0.5);
    ki_ = c.value("ki", 0.01);
    kd_ = c.value("kd", 0.01);
    min_out_ = c.value("min_sigma", 0.5);
    max_out_ = c.value("max_sigma", 2.0);
}

double PIDController::update(double error, int step) {
    // 1. 比例项
    double p_out = kp_ * error;

    // 2. 积分项 (带有简单的抗饱和处理)
    integral_ += error * step;
    double i_out = ki_ * integral_;

    // 3. 微分项 (带低通滤波)
    double derivative = (error - prev_error_) / step;
    filtered_d_ = d_filter_alpha * derivative + (1.0 - d_filter_alpha) * filtered_d_;
    double d_out = kd_ * filtered_d_;

    prev_error_ = error;

    // 4. 总输出并限制死区
    double output = p_out + i_out + d_out;
    return std::clamp(output, min_out_, max_out_);
}

void PIDController::reset() {
    integral_ = 0.0;
    prev_error_ = 0.0;
    filtered_d_ = 0.0;
}
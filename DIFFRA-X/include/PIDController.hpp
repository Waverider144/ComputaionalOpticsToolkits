#pragma once
#include "Controller.hpp"
#include <algorithm>
#include "external/json.hpp"

class PIDController : public Controller {
public:
    PIDController(const nlohmann::json& pid_config);

    double update(double error, int step = 1.0) override;
    void reset() override;

private:
    double kp_, ki_, kd_;
    double min_out_, max_out_;
    
    double integral_ = 0.0;
    double prev_error_ = 0.0;
    
    // 简单的低通滤波，防止微分项 D 瞬间爆炸
    double d_filter_alpha = 0.2; 
    double filtered_d_ = 0.0;
};
#pragma once
#include <external/json.hpp>
#include <algorithm>
#include <cmath>

class Controller {
public:
    virtual ~Controller() = default;

    virtual double update(double error, int step) = 0;
    // 重置积分项等状态
    virtual void reset() = 0;
};

class AnnealingController : public Controller {
public:
    AnnealingController(const nlohmann::json& c) {
        start_val = c.value("start", 2.0);
        end_val = c.value("end", 0.5);
        decay_rate = c.value("decay", 0.002);
    }
    double update(double error, int step) override {
        // T = T_min + (T_max - T_min) * exp(-k * step)
        return end_val + (start_val - end_val) * std::exp(-decay_rate * step);
    }
    void reset() override {}
private:
    double start_val, end_val, decay_rate;
};
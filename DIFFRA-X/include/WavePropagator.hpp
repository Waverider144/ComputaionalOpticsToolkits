#pragma once
#include <torch/torch.h>

class WavePropagator {
public:
    virtual ~WavePropagator() = default;
    virtual torch::Tensor propagate(torch::Tensor input, bool is_phase) = 0;
};
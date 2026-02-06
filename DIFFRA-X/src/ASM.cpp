#include "ASM.hpp"
#include <cmath>

ASM::ASM(int N, double wl, double dx, double z,torch::Device p_device) 
    : N(N), wl(wl), dx(dx), z(z),device(p_device) {
    precompute_transfer_function();
}

void ASM::precompute_transfer_function() {
    double df = 1.0 / (N * dx);
    double k = 2.0 * M_PI / wl;

    // 创建频率坐标系
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(device);
    auto lin = torch::arange(-N / 2, N / 2, options);
    auto f_grids = torch::meshgrid({lin, lin}, "ij");

    auto fx = f_grids[0].to(device) * df;
    auto fy = f_grids[1].to(device) * df;

    // 计算 kz = sqrt(k^2 - (2*pi*fx)^2 - (2*pi*fy)^2)
    auto k_sq = std::pow(k, 2);
    auto kx_sq = torch::pow(2.0f * M_PI * fx, 2);
    auto ky_sq = torch::pow(2.0f * M_PI * fy, 2);
    
    // 渐逝波处理：避免开方出现负数导致 NaN，物理上这部分波不传播
    auto argument = k_sq - kx_sq - ky_sq;
    auto mask = (argument > 0).to(torch::kFloat64);
    auto kz = torch::sqrt(torch::clamp(argument, 0.0));

    // 计算 H = exp(i * kz * z)
    auto phase = kz * z;
    auto H_real = torch::cos(phase) * mask;
    auto H_imag = torch::sin(phase) * mask;
    
    // 组合成复数张量并进行 fftshift
    H = torch::complex(H_real, H_imag).to(torch::kComplexFloat).to(device);
    H = fftshift(H); 
}

torch::Tensor ASM::propagate(torch::Tensor input, bool is_phase) {
    torch::Tensor field = is_phase ? 
        torch::complex(torch::cos(input), torch::sin(input)) : 
        input.to(torch::kComplexFloat);
    
    // input 可能已经是在 device 上的，确保 H 也在相同 device
    return torch::fft::ifft2(torch::fft::fft2(field) * H);
}

torch::Tensor ASM::fftshift(torch::Tensor x) {
    auto dim0 = x.size(0);
    auto dim1 = x.size(1);
    return torch::roll(x, {dim0 / 2, dim1 / 2}, {0, 1});
}
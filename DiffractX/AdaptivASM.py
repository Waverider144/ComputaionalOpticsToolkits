import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.interpolate import RegularGridInterpolator

def generate_u0(params, apertures):
    """生成源场 U0，支持常规孔径和连续相位涡旋"""
    L, N = params["L"], params["N"]
    x = np.linspace(-L/2, L/2, N)
    y = np.linspace(-L/2, L/2, N)
    X, Y = np.meshgrid(x, y)
    U0 = np.zeros((N, N), dtype=complex)
    
    for ap in apertures:
        # 获取基础参数
        x0, y0 = ap.get("x0", 0), ap.get("y0", 0)
        
        if ap["type"] == "vortex_circle":
            dX, dY = X - x0, Y - y0
            dist = np.sqrt(dX**2 + dY**2)
            phi = np.arctan2(dY, dX)
            l = ap.get("l", 1)
            
            # 这是一个 2D 矩阵
            amplitude_field = np.exp(1j * (l * phi + ap.get("phase", 0.0)))
            
            # 【关键修正】：使用掩码同时过滤索引和数据
            mask = dist < ap["radius"]
            U0[mask] = amplitude_field[mask] # 两边都是 1D 了，完美匹配

        elif ap["type"] == "circle":
            amplitude = np.exp(1j * ap.get("phase", 0.0))
            dist = np.sqrt((X - x0)**2 + (Y - y0)**2)
            U0[dist < ap["radius"]] = amplitude
            
        elif ap["type"] == "rect":
            amplitude = np.exp(1j * ap.get("phase", 0.0))
            mask_x = np.abs(X - x0) < (ap["width"] / 2)
            mask_y = np.abs(Y - y0) < (ap["height"] / 2)
            U0[mask_x & mask_y] = amplitude
            
    return U0, X, Y

def add_lens(U0, X, Y, params, lcfg):
    """给光场叠加透镜相位"""
    f, lr = lcfg["F"], lcfg["LensRadius"]
    k = 2 * np.pi / params["wavelength"]
    mask = np.sqrt(X**2 + Y**2) < lr
    # 核心：使用传入的 X, Y 矩阵计算二次相位
    lens_factor = np.exp(-1j * k / (2 * f) * (X**2 + Y**2))
    return U0 * lens_factor * mask

def run_asm(U0, params):
    """角谱法 (ASM)"""
    L, N, wavelength, z = params["L"], params["N"], params["wavelength"], params["z"]
    df = 1 / L
    A0 = fftshift(fft2(U0))
    fx = np.linspace(-N/2, N/2 - 1, N) * df
    fy = np.linspace(-N/2, N/2 - 1, N) * df
    FX, FY = np.meshgrid(fx, fy)
    k = 2 * np.pi / wavelength
    # 传递函数
    H = np.exp(1j * k * z * np.sqrt(1 - (wavelength * FX)**2 - (wavelength * FY)**2 + 0j))
    return ifft2(ifftshift(A0 * H))

def run_fresnel(U0, X, Y, params):
    """菲涅耳衍射 (带重采样)"""
    L, N, wavelength, z = params["L"], params["N"], params["wavelength"], params["z"]
    dx = L / N
    k = 2 * np.pi / wavelength
    U_pre = U0 * np.exp(1j * k / (2 * z) * (X**2 + Y**2))
    Uz_fourier = fftshift(fft2(ifftshift(U_pre)))
    L_prime = (wavelength * z) / dx
    x_prime = np.linspace(-L_prime/2, L_prime/2, N)
    y_prime = np.linspace(-L_prime/2, L_prime/2, N)
    Xp, Yp = np.meshgrid(x_prime, y_prime)
    term1 = np.exp(1j * k * z) / (1j * wavelength * z)
    term2 = np.exp(1j * k / (2 * z) * (Xp**2 + Yp**2))
    Uz = term1 * term2 * Uz_fourier * (dx**2)
    # 重采样回原始 L 网格
    if not np.isclose(L, L_prime):
        interp_real = RegularGridInterpolator((y_prime, x_prime), Uz.real, bounds_error=False, fill_value=0)
        interp_imag = RegularGridInterpolator((y_prime, x_prime), Uz.imag, bounds_error=False, fill_value=0)
        new_points = np.stack([Y.ravel(), X.ravel()], axis=-1)
        Uz = (interp_real(new_points) + 1j * interp_imag(new_points)).reshape((N, N))
    return Uz

def adaptive_solver(params, apertures, lens_cfg=None):
    """自适应求解器"""
    # 1. 初始化源场
    U0, X, Y = generate_u0(params, apertures)

    # 2. 挂载透镜 (注意：这里统一使用传入的 params 和 lens_cfg)
    if lens_cfg is not None:
        print(f"Adding lens with f = {lens_cfg['F']}m")
        U0 = add_lens(U0, X, Y, params, lens_cfg)
    
    # 3. 计算 Fn 决定算法
    a = max([ap.get("radius", 0) for ap in apertures])
    Fn = (a**2) / (params["wavelength"] * params["z"])
    
    if Fn > 1.0:
        print(f"Fn = {Fn:.2f}: Using ASM")
        Uz = run_asm(U0, params)
    else:
        print(f"Fn = {Fn:.2f}: Using Fresnel")
        Uz = run_fresnel(U0, X, Y, params)
        
    # 4. 绘图
    intensity = np.abs(Uz)**2
    intensity /= (np.max(intensity) + 1e-12) # 归一化
    
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.imshow(np.abs(U0)**2, extent=[-params["L"]/2, params["L"]/2, -params["L"]/2, params["L"]/2])
    plt.title("Source Field")
    
    plt.subplot(122)
    # 使用 log 尺度看清干涉条纹和暗纹
    plt.imshow(np.log10(intensity + 1e-5), extent=[-params["L"]/2, params["L"]/2, -params["L"]/2, params["L"]/2], cmap='jet')
    plt.title(f"Result (z={params['z']}m, Phase Diff=PI)")
    plt.colorbar(label="log10(Intensity)")
    plt.show()

# --- 物理配置区 (单位：米) ---
MY_CONFIG = {
    "wavelength": 633e-9, 
    "L": 0.01,    # 放大观测窗到 10mm
    "N": 1024, 
    "z": 0.1       
}

MY_APERTURES = [
    {
        "type": "vortex_circle", 
        "radius": 0.0005,  # 半径 0.5mm
        "x0": 0.0015, "y0": 0, 
        "l": 3, "phase": 0.0
    },
    {
        "type": "vortex_circle", 
        "radius": 0.0005, 
        "x0": -0.0015, "y0": 0, 
        "l": -3, "phase": np.pi
    }
    # {
    #     "type": "rect", 
    #     "width": 0.00001, "height": 0.00001, 
    #     "x0": 0.001, "y0": 0.001,          # 位于中心下方
    #     "phase": (0.5 * np.pi)                   # 依然给个反相，看中间那条横着的黑线
    # },
    # {
    #     "type": "rect", 
    #     "width": 0.00001, "height": 0.00001, 
    #     "x0": -0.001, "y0": 0.001,          # 位于中心下方
    #     "phase": np.pi                   # 依然给个反相，看中间那条横着的黑线
    # },
    # {
    #     "type": "rect", 
    #     "width": 0.00001, "height": 0.00001, 
    #     "x0": 0.001, "y0": -0.001,          # 位于中心下方
    #     "phase": (1.5 *np.pi)                   # 依然给个反相，看中间那条横着的黑线
    # }
]

MY_LENS = {"F": 0.1, "LensRadius": 0.05}

# 执行
adaptive_solver(MY_CONFIG, MY_APERTURES, MY_LENS)
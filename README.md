# Computational Optics Toolkits

**Computational Optics Toolkits (COT)** is a comprehensive software library independently developed to provide high-performance computational support for optical research and engineering. By integrating core algorithms of modern computational optics, COT streamlines the workflow from physical modeling to numerical simulation, offering a flexible and scalable platform for researchers and engineers.

## Table of Contents

DiffractX
DIFFRA-X

## DiffractX

DiffractX is a diffraction simulation tool combined Angular Spectrum Method(ASM) and Fresnel Method with hard switch with criterion of Fresnel Number(Fn) to achieve full-scale simulation. Three categories of sources are provided: Circle, Rectangle and Vortex. Written in Python
![DiffractX](https://raw.githubusercontent.com/Waverider144/Assets/main/DiffractX.png)

## DIFFRA-X

<div align="center">
  <img src="https://github.com/Waverider144/Assets/raw/refs/heads/main/testouput_H%20A%20R%20V%20E%20Y.png" width="400" alt="DIFFRA-X">
  <p><i></i></p>
</div>

**DIFFRA-X** is a high-performance computational optics platform implemented in C++ utilizing the LibTorch (PyTorch C++ API) backend. It is specifically engineered to address complex phase retrieval and digital holographic reconstruction tasks through differentiable physics-based modeling.
<table border="0">
  <tr>
    <td width="50%">
      <img src="https://raw.githubusercontent.com/Waverider144/Assets/main/benchmark.jpeg" alt="Benchmark Origin" style="width:100%;">
      <p align="center"><b>USAF Benchmark Origin</b></p>
    </td>
    <td width="50%">
      <img src="https://raw.githubusercontent.com/Waverider144/Assets/main/recon_step_29980.png" alt="Design Outcome" style="width:100%;">
      <p align="center"><b>Design Outcome</b></p>
    </td>
  </tr>
</table>

### 1. Theoretical Foundation

The core physical engine is predicated on **Scalar Diffraction Theory**, employing the **Angular Spectrum Method (ASM)** for rigorous wave-field propagation.

#### Physical Model

Given an initial complex field distribution $U_0​(x,y)$, the field $U_z​(x,y)$ at a propagation distance z is computed via:

$$U_z = \mathcal{F}^{-1}\{\mathcal{F} {U_0} \cdot{H}\}$$

The transfer function $H(f_x​,f_y​)$ is defined as
$$H = exp(j \frac{2 \pi z}{\lambda}\sqrt{1-(\lambda f_x)^2-(\lambda f_y)^2}$$


#### Optimization Strategy

The framework embeds the diffraction process within an automatic differentiation engine. The phase distribution is optimized by minimizing the Mean Squared Error (MSE) between the synthesized intensity and the target constraints:

-   **Adaptive Physical Constraints**: Incorporates a dynamic σ parameter to modulate the strength of physical priors during the optimization trajectory.
    
-   **Automated Control Loops**: Supports **Proportional-Integral-Derivative (PID)** control and **Exponential Annealing(Recommended)** schedules to regulate the solver state and prevent convergence into local minima.
    

----------

### 2. System Architecture

The platform adopts a modular architecture to decouple physical simulation from optimization heuristics:

Module

Functional Description

**`Config`**

Manages JSON-based serialization of global parameters (N,λ, pitch, distance).

**`ASM`**

The core diffraction operator implementing FFT-based wave propagation.

**`DiffEngine`**

Facilitates forward propagation and adjoint-based gradient computation.

**`PhaseOptimizer`**

Orchestrates the optimization loop, synchronizing the Adam optimizer with the feedback controllers.

**`Controller`**

Abstract base for heuristic scheduling (PID and Annealing implementations).

**`ImageManager`**

IO subsystem supporting automated tensor-to-image conversion and disk quota management.

Export to Sheets

----------

### 3. Implementation and Usage

#### Prerequisites

-   **LibTorch**: Version 2.0+ (CUDA-enabled recommended).
    
-   **OpenCV**: For image serialization and real-time visualization.
    
-   **C++ Standard**: C++17 or higher.
    

#### Execution Pipeline

1.  **Configuration**: Define parameters in `config.json`:
    
    JSON
    
    ```
    {
        "physics": { "N": 512, "wavelength": 532e-9, "pixel_size": 8e-6, "distance": 0.05 },
        "optimizer": { "sigma_mode": "pid", "learning_rate": 0.01, "max_steps": 1000 },
        "pid": { "kp": 0.2, "ki": 0.01, "kd": 0.0, "min_sigma": 0.5, "max_sigma": 2.5 }
    }
    
    ```
    
2.  **Build and Run**:
    
    Bash
    
    ```
    mkdir build && cd build
    cmake ..
    make -j
    ./diffra_x
    
    ```
    

#### Core API Integration

C++

```
// Initialize the diffraction engine
auto asm_phy = std::make_shared<ASM>(config.N, config.wavelength, config.pixel_size, config.distance, device);

// Instantiate the phase retrieval solver
auto solver = std::make_unique<PhaseOptimizer>(diff_engine, target_tensor, config);

// Automated output management with disk cleaning
ImageManager::save_and_limit(recon_tensor, current_step, config.output_dir, 1000);

```

----------

## 4. Key Features

-   **Automated Disk Maintenance**: Implements a Least-Recently-Used (LRU) style file management system, maintaining a fixed buffer of the most recent 1,000 optimization frames.
    
-   **Self-Adaptive Constraints**: Utilizes closed-loop PID control to adjust physical constraints based on the instantaneous loss decay rate.
    
-   **Numerical Robustness**: Integrated NaN-detection mechanisms to monitor gradient stability and ensure safe termination upon numerical divergence.

## Change Log



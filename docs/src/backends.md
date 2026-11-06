# Backends

JustPIC uses KernelAbstractions backends directly. The CPU path is
`JustPIC.CPU`; CUDA, AMDGPU, and Metal backends are activated by loading their
respective packages:

| Hardware | Package | Backend |
| --- | --- | --- |
| CPU | — | `JustPIC.CPU` |
| Nvidia GPU | `CUDA` | `CUDA.CUDABackend` |
| AMD GPU | `AMDGPU` | `AMDGPU.ROCBackend` |
| Apple GPU | `Metal` | `Metal.MetalBackend` |

Allocate backend arrays with `TA(backend)` and pass the same backend to
`init_particles`:

```julia
using JustPIC
using CUDA

backend = CUDA.CUDABackend   # or JustPIC.CPU, AMDGPU.ROCBackend, Metal.MetalBackend
T = TA(backend)(zeros(Float32, 32, 32))
particles = init_particles(backend, 16, 24, 8, grid_vx, grid_vy)
```

The high-level particle, interpolation, and advection routines then dispatch
on the arrays stored in `particles`:

```julia
V = TA(backend).(V_cpu)
advection!(particles, RungeKutta2(), V, dt)
```

Allocating every array through `TA(backend)` keeps a script backend-generic:
switching hardware means changing the `backend` line only. Use `Float32` grids
and fields when targeting Metal, which does not support `Float64`.

For mixed CPU/GPU simulations, keep the JustPIC backend on the CPU and copy
only the fields needed by GPU calculations. See [Mixed CPU and GPU](mixed_CPU_GPU.md).

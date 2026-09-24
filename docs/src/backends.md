# Backends

JustPIC uses KernelAbstractions backends directly. The CPU path is
`JustPIC.CPU`; CUDA, AMDGPU, and Metal backends are activated by loading their
respective packages.

Allocate backend arrays with `TA(backend)` and pass the same backend to
`init_particles`:

```julia
using JustPIC

backend = JustPIC.CPU
T = TA(backend)(zeros(Float32, 32, 32))
particles = init_particles(backend, 16, 24, 8, grid_vx, grid_vy)
```

The high-level particle, interpolation, and advection routines then dispatch
on the arrays stored in `particles`:

```julia
V = TA(backend).(V_cpu)
advection!(particles, RungeKutta2(), V, dt)
```

Keep shared code backend-generic: do not branch on `Array`, `CuArray`, or
vendor-specific array types. Use `Float32` when targeting Metal, which does
not support `Float64` kernels.

For mixed CPU/GPU simulations, keep the JustPIC backend on the CPU and copy
only the fields needed by GPU calculations. See [Mixed CPU and GPU](mixed_CPU_GPU.md).

## API

```@docs
TA
launch!
to_cpu
```

# Mixed CPU and GPU computations

If GPU memory is a limiting factor for your computation, it may be preferable to
carry out particle operations on the CPU while the rest of the solver runs on the
GPU. This involves four steps.

## 1. Set the JustPIC backend to the CPU

At the top of the script, set the JustPIC backend to the CPU; other packages may
still run their own GPU work:

```julia
using JustPIC
const backend = JustPIC.CPU
```

## 2. Allocate the mirrored arrays

At the memory-allocation stage, allocate GPU copies of the particle-derived CPU
fields. For example, phase ratios on mesh vertices:

```julia
using CUDA

phv_GPU = cell_array(CUDA.CUDABackend, 0.0, (N_phases,), (nx + 1, ny + 1, nz + 1))
```

where `N_phases` is the number of material phases and
`cell_array(CUDA.CUDABackend, ...)` allocates a GPU-backed `CellArray`.

Similarly, allocate CPU copies of the GPU velocity field:

```julia
V_CPU = (
    x = zeros(nx + 1, ny + 2, nz + 2),
    y = zeros(nx + 2, ny + 1, nz + 2),
    z = zeros(nx + 2, ny + 2, nz + 1),
)
```

## 3. Copy particle-derived fields to the GPU

At each time step, transfer the particle-derived information from CPU to GPU
memory. For example, the vertex phase proportions:

```julia
phv_GPU.data .= CuArray(phase_ratios.vertex).data
```

## 4. Copy the velocity field to the CPU

At each time step, once the velocity computation on the GPU is finished,
transfer the velocity to the CPU:

```julia
V_CPU.x .= TA(backend)(V.x)
V_CPU.y .= TA(backend)(V.y)
V_CPU.z .= TA(backend)(V.z)
```

Because `backend == JustPIC.CPU`, `TA(backend)` converts the velocity fields to
plain CPU arrays. Advection then uses the usual JustPIC API:

```julia
advection!(particles, RungeKutta2(), values(V_CPU), Δt)
```

# Mixed CPU and GPU computations

If GPU memory is a limiting factor for your computation, it may be preferable to carry out particle operations on the CPU rather than on the GPU.
This involves basically four steps:

1. *At the top of the script*. The JustPIC backend must be set to CPU, while other packages may still run their own GPU work:
```julia
const backend = JustPIC.CPU 
```

2. *At memory allocation stage*. A copy of relevant CPU arrays must be allocated on the GPU memory. For example, phase ratios on mesh vertices:
```julia
using JustPIC
using CUDA

phv_GPU = cell_array(CUDA.CUDABackend, 0.0, (N_phases,), (nx + 1, ny + 1, nz + 1))
```
where `N_phases` is the number of different material phases and
`cell_array(CUDA.CUDABackend, ...)` allocates a GPU-backed `CellArray`.

Similarly, GPU arrays must be copied to CPU memory:
```julia
V_CPU = (
    x = zeros(nx+1, ny+2, nz+2),
    y = zeros(nx+2, ny+1, nz+2),
    z = zeros(nx+2, ny+2, nz+1),
)
```
where `zeros()` allocates on the CPU memory.

3. *At each time step*. The particles are stored in CPU memory. It is hence necessary to transfer some information from the CPU to the GPU memory. For example, here's a transfer of phase proportions:

```julia
phv_GPU.data .= CuArray(phase_ratios.vertex).data
```

4. *At each time step*. Once velocity computations are finalized on the GPU, they need to be transferred to the CPU:

```julia
V_CPU.x .= TA(backend)(V.x)
V_CPU.y .= TA(backend)(V.y)
V_CPU.z .= TA(backend)(V.z)
```
Advection can then be applied by calling the `advection()` function:

```julia
advection!(particles, RungeKutta2(), values(V), Δt)
```

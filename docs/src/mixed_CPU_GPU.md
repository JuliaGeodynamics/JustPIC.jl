# Mixed CPU and GPU computations

If GPU memory is a limiting factor for your computation, it may be preferable to carry out particle operations on the CPU rather than on the GPU.
This involves basically 4 steps:

1) *At the top of the script*. The JustPIC backend must be set to CPU, in contrast with other employed packages (e.g. ParallelStencil):
```julia
const backend = JustPIC.CPUBackend 
```

2) *At memory allocation stage*. A copy of relevant CPU arrays must be allocated on the GPU memory. For example, phase ratios on mesh vertices:
```julia
phv_GPU = @zeros(nx+1, ny+1, nz+1, celldims=(N_phases))
```
where `N_phases` is the number of different material phases and `@zeros()` allocates on the GPU.

Similarly, GPU arrays must be copied to CPU memory:
```julia
V_CPU = (
        x      = zeros(Nc.x+1, Nc.y+2, Nc.z+2),
        y      = zeros(Nc.x+2, Nc.y+1, Nc.z+2),
        z      = zeros(Nc.x+2, Nc.y+2, Nc.z+1),
    )
```
where `zeros()` allocates on the CPU memory.

4) *At each time step*. The particle will be stored on the CPU memory. It is hence necessary to transfer some information from the CPU to the GPU memory. For example, here's a transfer of phase proportions:

```julia
phv_GPU.data .= CuArray(phase_ratios.vertex).data
```
!!! we explicitly write `CuArray` - would be better to have something more explicit like `GPUArray` - is there such a thing?

5) *At each time step*. Once velocity computation are finalised on the GPU, they need to be transferred to the CPU:

```julia
V_CPU.x .= TA(backend)(V.x)
V_CPU.y .= TA(backend)(V.y)
V_CPU.z .= TA(backend)(V.z)
```
Advection can then be applied by calling the `advection()` function:

```julia
advection!(particles, RungeKutta2(), values(V), (grid_vx, grid_vy, grid_vz), Δt)
```

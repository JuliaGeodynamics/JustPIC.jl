# Mixed CPU and GPU computations

If GPU memory is a limiting factor, it can be useful to keep particle storage and
particle operations on the CPU while solving mesh-based equations on the GPU.
This workflow has two important consequences:

- JustPIC particle containers should be allocated with `CPUBackend`.
- Mesh fields that are consumed by the GPU solver must be copied explicitly
  between CPU and GPU memory.

## 1. Choose the particle backend

At the top of the script, keep JustPIC particles on the CPU even if other
packages use a GPU backend:

```julia
const backend = JustPIC.CPUBackend 
```

## 2. Allocate matching CPU and GPU buffers

At allocation time, create GPU buffers for the mesh quantities needed by the
solver. For example, phase ratios on mesh vertices can be mirrored on the GPU:

```julia
phv_GPU = @zeros(nx+1, ny+1, nz+1, celldims=(N_phases))
```

Here `N_phases` is the number of material phases and `@zeros` allocates on the
active `ParallelStencil` backend.

You also need CPU-side velocity buffers for particle advection:

```julia
V_CPU = (
    x = zeros(nx+1, ny+2, nz+2),
    y = zeros(nx+2, ny+1, nz+2),
    z = zeros(nx+2, ny+2, nz+1),
)
```

## 3. Copy particle-derived fields to the GPU

At each time step, particle-derived fields that feed the GPU solver must be
copied from CPU storage to GPU storage. For example:

```julia
phv_GPU.data .= CuArray(phase_ratios.vertex).data
```

Use the array constructor that matches your active accelerator backend, such as
`CuArray` for CUDA or `ROCArray` for AMDGPU.

## 4. Copy solver velocities back to the CPU

After the GPU velocity solve, transfer the staggered velocity components back to
CPU arrays:

```julia
V_CPU.x .= TA(backend)(V.x)
V_CPU.y .= TA(backend)(V.y)
V_CPU.z .= TA(backend)(V.z)
```

Advection can then be applied with the usual JustPIC API:

```julia
advection!(particles, RungeKutta2(), values(V), Δt)
```

Because `backend == CPUBackend`, `TA(backend)` converts the velocity fields to
plain CPU arrays before particle advection.

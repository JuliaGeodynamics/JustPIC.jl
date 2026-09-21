# Public API

The reference is organized by workflow. Each manual page documents the symbols
belonging to its topic; this page collects the remaining backend and grid
helpers and provides a searchable index of everything.

## Where to find each group

| Topic | Page |
| --- | --- |
| Particle containers, initialization, advection, maintenance, phase ratios, subgrid diffusion | [Particles](particles.md) |
| Cell storage, allocation and halo exchange | [CellArrays](CellArrays.md) |
| Grid/particle and centroid transfers | [Interpolations](interpolations.md) |
| Velocity reconstruction schemes | [Velocity interpolation](velocity_interpolation.md) |
| Marker-chain surface tracking | [Marker chain](marker_chain.md) |
| Marker-surface (3D free surface) tracking | [Marker surface](marker_surface.md) |
| Checkpointing and restart | [I/O](IO.md) |

## Backends

JustPIC dispatches on KernelAbstractions backend types directly — `CPU` for the CPU
path and the vendor backends `CUDA.CUDABackend`, `AMDGPU.ROCBackend`,
`Metal.MetalBackend`, introduced by the respective package extensions. Load the
vendor package to activate its extension; JustPIC defines no backend tags of its own.

`TA(backend)` selects the plain array type for a backend, and `launch!` runs a
KernelAbstractions kernel on it, synchronizing before returning.

```@docs
TA
launch!
```

## Grid helpers

```@docs
add_periodic_ghost_nodes
```

## Container conversion

`to_cpu` copies arrays and JustPIC cell arrays to CPU storage. Pass a numeric
type to convert the stored values during the copy; tuples are converted
recursively.

```@docs
to_cpu
```

## Index

```@index
```

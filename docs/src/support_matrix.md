# Support matrix

This matrix records current release support. `Tested` means covered by the
repository test suite or a documented example. `Experimental` means the path
exists but lacks complete release coverage. `Unsupported` means do not rely on
the combination.

| Subsystem | 2D | 3D | CPU | CUDA | AMDGPU | Metal | Float32 | Float64 | Uniform grid | Refined grid | MPI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Material particles | Tested | Tested | Tested | Tested | Tested | Tested | Tested | Tested | Tested | Tested | Experimental |
| Passive markers | Tested | Experimental | Tested | Experimental | Experimental | Experimental | Tested | Tested | Tested | Experimental | Unsupported |
| Marker chains | Tested | Unsupported | Tested | Tested | Tested | Tested | Tested | Tested | Tested | Tested | Experimental |
| Marker surfaces | Unsupported | Tested | Tested | Tested | Tested | Tested | Tested | Unsupported on Metal | Tested | Experimental | Tested |
| Phase ratios | Tested | Tested | Tested | Tested | Tested | Tested | Tested | Tested | Tested | Experimental | Experimental |
| Subgrid diffusion | Tested | Tested | Tested | Tested | Tested | Tested | Tested | Tested | Tested | Experimental | Experimental |
| Checkpoint/restart | Tested | Tested | Tested | Tested | Tested | Tested | Tested | Tested | Tested | Experimental | Experimental |

## Qualification rules

- Metal supports `Float32`; Metal does not support `Float64`.
- Refined-grid support means coordinate-vector grids, not only `LinRange` grids.
- GPU MPI requires a compatible GPU-aware MPI configuration.
- `Experimental` combinations may work, but are not release guarantees.
- The matrix must change with the test suite and release checklist.

# Backend Portability (CUDA, AMDGPU, Metal)

All code must run unchanged on the CPU, CUDA, AMDGPU, and Metal backends. Metal has no
`Float64`, so every code path must also work with `Float32` data.

## Element-type genericity

- Never hard-code `Float64` literals or types in kernels or the helpers they call. Derive
  the type from the data: `T = eltype(x)`, then `zero(T)`, `one(T)`, `T(0.5)`,
  `inv(T(2))`.
- Integer literals (`2`, `1`) are fine in arithmetic with floats; `0.5`, `1e-10`, `π`
  written as bare literals promote to `Float64` — wrap them in `T(...)`.
- Avoid functions that promote silently (`float(::Int)`, `sqrt(2)`, `rand()` with no
  type); pass `T` explicitly.
- Tolerances and constants that depend on precision should use `eps(T)`, not a fixed
  `1e-12`.

## Kernel rules

- Use KernelAbstractions `@kernel`/`@index` and launch through `launch!` with
  `ka_backend(x)`; no branching on `Array`/`CuArray`/`ROCArray`/`MtlArray` in shared code.
- Kernels must be allocation-free and type-stable (use tuples and `SVector`s, not temporary
  arrays). Dynamic dispatch or boxing fails on every GPU backend.
- No scalar indexing of device arrays outside kernels; move data to the host with
  `Array(...)` when a scalar is needed.

## Checking

- Run new code on the CPU with `Float32` inputs as a stand-in for Metal: results must
  have `Float32` eltype (no silent promotion to `Float64`) and pass the same tests with
  tolerances scaled to `eps(Float32)`.
- `@code_warntype` on a kernel's CPU instantiation catches most
  instabilities before they surface as GPU compilation errors.

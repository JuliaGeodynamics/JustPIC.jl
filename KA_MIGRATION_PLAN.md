# Migration plan: ParallelStencil.jl → KernelAbstractions.jl

Status: **in progress / CPU path green and clean**
Scope: entire package (`src/`, `ext/`, `test/`, `scripts/`, `docs/`)

Progress snapshot (2026-07-02, from the current worktree):
- [x] `KernelAbstractions` has been added to `Project.toml`; direct `ParallelStencil` and `Atomix`
      dependencies have been removed.
- [x] `src/launch.jl` now exists with `ka_backend`, synchronous `launch!`, and backend-aware
      `cell_array` scaffolding.
- [x] `common.jl` includes `launch.jl`, and the CPU `_2D`/`_3D` modules import KA's bare
      `@index` while qualifying CellArraysIndexing element access as `CAI.@index`.
- [x] The CellArraysIndexing qualification pass is mostly done in `src/` and `test/test_CellArrays.jl`
      (`rg "CAI\\.@index" src test` currently finds 169 sites).
- [x] `src/Interpolations/` has been ported to KA launches/kernels.
- [x] Particle advection kernels `advection.jl`, `advection_LinP.jl`, and `advection_MQS.jl`
      have been ported to KA.
- [x] The CPU `src/` path no longer uses active ParallelStencil macros or stale migration comments.
- [x] `test/test_2D.jl`, `test/test_3D.jl`, and `test/Project.toml` no longer depend on
      ParallelStencil; CPU tests now use KA helpers directly.
- [x] CUDA/AMDGPU extensions no longer contain active ParallelStencil macros/imports; they still keep
      the old `_2D`/`_3D` forwarding layer pending a later collapse.
- [x] Script/doc active ParallelStencil examples have been ported or removed.
- [x] CPU-side migration lint is now part of the test suite.
- [x] CPU timer and MPI smoke scripts are runnable in local CPU mode.
- [x] Public docs/docstrings now describe backend-qualified tags and `cell_array` allocation
      instead of legacy ParallelStencil allocation macros.
- [ ] GPU runtime validation, GPU benchmarks, and release/optional cleanup are not yet migrated.

## Checkpoint log

- 2026-07-02 19:40 CEST — area: baseline; validation:
  `julia --project=. -e 'using JustPIC; println("JustPIC loaded")'` => pass; remaining:
  49 `@parallel_indices`, 44 `@parallel`, 13 `@fill`; notes: CPU source still depends on
  ParallelStencil, but the current partial KA branch imports successfully.
- 2026-07-02 19:43 CEST — area: `Particles/Advection/backtracking*` done; validation:
  `JULIA_JUSTPIC_BACKEND=CPU julia --project=. test/test_2D.jl` => pass; remaining:
  43 `@parallel_indices`, 41 `@parallel`, 13 `@fill`; notes: KA offset ndrange uses
  `I0 = @index(Global, NTuple); I = I0 .+ 1` to satisfy KA CPU codegen.
- 2026-07-02 19:49 CEST — area: particle allocation/move/injection utilities done;
  validation: `JULIA_JUSTPIC_BACKEND=CPU julia --project=. test/test_2D.jl` => pass;
  remaining: 29 `@parallel_indices`, 32 `@parallel`, 5 `@fill`; notes:
  `cell_array` is now backend-explicit, `ka_backend(::CellArray)` dispatches through `.data`,
  and public injection APIs keep their names while internal KA kernels use `_kernel!` suffixes.
- 2026-07-02 19:59 CEST — area: CPU `src/` Phase 1 done; validation:
  `julia --project=. -e 'using Pkg; Pkg.test(; test_args=["--backend=CPU"])'` => pass;
  remaining: 0 active `@parallel_indices`, 0 active `@parallel`, 0 active `@fill`; notes:
  only explanatory PS references remain in `src/launch.jl`; Aqua temporarily ignores
  `ParallelStencil` as an intentional dependency until GPU extensions/scripts/docs are migrated.
- 2026-07-02 20:13 CEST — area: CPU test path cleanup done; validation:
  `julia --project=. -e 'using Pkg; Pkg.test(; test_args=["--backend=CPU"])'` => pass;
  remaining: 0 `@parallel_indices`, 6 `@parallel`, 0 `@fill`; notes: remaining active
  ParallelStencil footprint is confined to CUDA/AMDGPU extensions and the main dependency entry;
  `test/Project.toml` now depends on KernelAbstractions instead of ParallelStencil.
- 2026-07-02 20:20 CEST — area: CUDA/AMDGPU extension PS macro cleanup partial;
  validation: `julia --project=. -e 'Meta.parseall(read("ext/JustPICCUDAExt.jl", String)); Meta.parseall(read("ext/JustPICAMDGPUExt.jl", String)); println("extension files parsed")'`
  => pass; `julia --project=. -e 'using Pkg; Pkg.test(; test_args=["--backend=CPU"])'`
  => pass; remaining: 4 `@parallel_indices`, 6 `@parallel`, 1 `@fill`; notes: `src/`,
  `test/`, and `ext/` active PS macro/import scan is clean; remaining macro sites are in
  scripts/docs only. CUDA/AMDGPU packages are not installed in this environment, so extension
  runtime loading/GPU execution is still unvalidated here.
- 2026-07-02 20:24 CEST — area: scripts/docs and dependency cleanup done; validation:
  `julia --project=. -e 'for f in ["scripts/pureshear_ALE.jl", "scripts/pureshear_ALE_restart.jl", "scripts/rotating_marker_chain.jl", "scripts/rotating_marker_chain_SML.jl", "scripts/rotating_marker_chain_SML_topography.jl", "scripts/temperature_advection3D_MPI.jl"]; Meta.parseall(read(f, String)); end; println("scripts parsed")'`
  => pass; `julia --project=. -e 'using Pkg; Pkg.test(; test_args=["--backend=CPU"])'`
  => pass; remaining: 0 `@parallel_indices`, 0 `@parallel`, 0 `@fill`; notes:
  `Project.toml` no longer depends on `ParallelStencil` or direct `Atomix`, Aqua stale-deps
  passes without ignores, and the remaining PS mentions are comments/checkpoint history only.
  CUDA/AMDGPU runtime validation remains blocked locally because those weakdeps are not installed.
- 2026-07-02 20:39 CEST — area: CPU lint/timer/MPI smoke partial; validation:
  `julia --project=. -e 'using Pkg; Pkg.test(; test_args=["--backend=CPU"])'` => fail;
  remaining: 0 `@parallel_indices`, 0 `@parallel`, 0 `@fill`; notes: new migration lint caught
  two stale legacy-macro references in `src/launch.jl` docstrings; timer smoke
  (`JUSTPIC_TIMER_N=32`, `JUSTPIC_TIMER_NITER=1`) and MPI smoke (`JUSTPIC_MPI_CI_N=16`,
  `JUSTPIC_MPI_CI_NITER=1`) both passed.
- 2026-07-02 20:42 CEST — area: CPU lint/timer/MPI smoke done; validation:
  `julia --project=. -e 'using Pkg; Pkg.test(; test_args=["--backend=CPU"])'` => pass;
  remaining: 0 `@parallel_indices`, 0 `@parallel`, 0 `@fill`; notes: migration lint now passes,
  active macro/import scan is clean, `scripts/temperature_advection_timer.jl` no longer needs
  TimerOutputs or stale REPL snippets. Local post-port timer medians from iterations 2-5:
  1 thread advect/move/injection/p2g/g2p ≈ 0.096/0.033/0.068/0.031/0.004 s; 4 threads ≈
  0.039/0.017/0.030/0.014/0.002 s; 8 threads ≈ 0.027/0.014/0.022/0.009/0.001 s.
- 2026-07-02 21:08 CEST — area: CPU cleanup sweep done; validation:
  `julia --project=. -e 'using Pkg; Pkg.test(; test_args=["--backend=CPU"])'` => pass;
  remaining: 0 `@parallel_indices`, 0 `@parallel`, 0 `@fill`; notes: deleted stale
  commented-out legacy launch snippets, widened `Migration lint` to scan `test/`, and ran
  `Pkg.resolve()` for the ignored local manifest so local CPU test runs no longer warn that
  the manifest is stale.
- 2026-07-02 21:15 CEST — area: docs/docstrings CPU cleanup done; validation:
  `julia --project=. -e 'using JustPIC; @assert isdefined(JustPIC, :CUDABackend); println("loaded")'`
  => pass; `julia --project=. -e 'using Pkg; Pkg.test(; test_args=["--backend=CPU"])'`
  => pass; remaining: 0 `@parallel_indices`, 0 `@parallel`, 0 `@fill`; notes: README/docs
  now use `JustPIC.*Backend` names and `cell_array` examples; core now defines the
  `CUDABackend` tag imported by the CUDA extension, while GPU runtime validation remains later.
- 2026-07-02 21:20 CEST — area: docs build validation done; validation:
  `julia --project=docs docs/make.jl` => pass; remaining: 0 `@parallel_indices`,
  0 `@parallel`, 0 `@fill`; notes: `docs/make.jl` now loads this checkout before the docs
  environment registry dependency, so API doc bindings validate against the working tree.
  Documenter still reports the existing non-fatal backlog of uncatalogued docstrings.

---

## 1. Why migrate, and what we get

JustPIC uses ParallelStencil (PS) only as a *kernel launcher*, not as a stencil DSL: there are no
`@all/@inn/@av` stencil macros, no `@hide_communication`, no shared-memory usage anywhere in the
codebase. Everything PS does for us is expressible in KernelAbstractions (KA) directly:

| PS feature actually used | Where | KA replacement |
|---|---|---|
| `@init_parallel_stencil(Backend, Float64, N)` | `src/JustPIC_CPU.jl`, both `ext/` files (6 call sites) | nothing — backend is a runtime value obtained from arrays |
| `@parallel_indices (i…) function …` kernel definitions | ~60 kernels across 33 files in `src/` | `@kernel function …` + `@index(Global, NTuple)` |
| `@parallel (ranges) kernel!(args…)` launches | ~50 call sites | `kernel!(backend)(args…; ndrange = …)` + explicit `synchronize` |
| `@fill(x, ni…, celldims=…, eltype=…)` CellArray allocation | `src/Particles/particles_utils.jl`, `src/MarkerChain/init.jl`, `src/Physics/subgrid_diffusion.jl` | direct `CellArrays` constructors + `fill!` via a small helper |
| `CuCellArray` / `ROCCellArray` type aliases (defined by PS init) | both `ext/` files | `CellArrays.@define_CuCellArray()` / `@define_ROCCellArray()` |
| Per-backend `@myatomic` macro (Atomix / `CUDA.@atomic` / `AMDGPU.@atomic`) | defined in each backend module, used in `src/PassiveMarkers/particle_to_grid.jl` | single `KernelAbstractions.@atomic` (Atomix-based, works on all backends) |

**The structural payoff** is bigger than the dependency swap. The whole reason
`src/common.jl` is textually `include`d **six times** (2D/3D × CPU/CUDA/AMDGPU) and the reason
`ext/JustPICCUDAExt.jl` / `ext/JustPICAMDGPUExt.jl` are ~1,180 lines each is that
`@init_parallel_stencil` bakes backend + dimension into a module at macro-expansion time, forcing
one module per (backend, dim) pair plus a hand-written forwarding layer
(`JustPIC._2D.advection!(::Particles{CUDABackend}, …) = advection!(…)`, repeated for every public
function). With KA, kernels are backend-generic and dimension-generic; the extensions shrink to
host↔device conversions (`CuArray(particles)` etc.) plus `TA`/`CA` definitions — roughly 150 lines
each. As a bonus, Metal and oneAPI backends become nearly free to add later
(`CellArrays.@define_MtlCellArray` already exists).

**Non-goals:** no public API changes (`JustPIC._2D` / `JustPIC._3D` namespaces, function names and
signatures stay), no algorithm changes, no MPI/ImplicitGlobalGrid changes
(`update_cell_halo!` operates on `.data` arrays and is PS-independent).

---

## 2. Translation reference

### 2.1 Kernel definition

Before (dimension-agnostic form, e.g. `src/Particles/Advection/advection.jl:54`):

```julia
@parallel_indices (I...) function advection_kernel!(p, method, V, index, grid_vi, local_limits, dxi, dt)
    for ipart in cellaxes(index)
        doskip(index, ipart, I...) && continue
        pᵢ = get_particle_coords(p, ipart, I...)
        pᵢ_new = advect_particle(method, pᵢ, V, grid_vi, local_limits, dxi, dt, I)
        for k in 1:N
            @index p[k][ipart, I...] = pᵢ_new[k]
        end
    end
    return nothing
end
```

After:

```julia
@kernel inbounds = true function advection_kernel!(p, method, V, index, grid_vi, local_limits, dxi, dt)
    I = @index(Global, NTuple)
    for ipart in cellaxes(index)
        doskip(index, ipart, I...) && continue
        pᵢ = get_particle_coords(p, ipart, I...)
        pᵢ_new = advect_particle(method, pᵢ, V, grid_vi, local_limits, dxi, dt, I)
        for k in 1:N
            CAI.@index p[k][ipart, I...] = pᵢ_new[k]   # ⚠ see §3.1 — CAI macro is qualified
        end
    end
end
```

Notes:
- `@index(Global, NTuple)` returns an `NTuple{N}` where `N` is the ndrange dimensionality, so the
  existing dimension-agnostic `(I...)` kernels translate 1:1. Kernels written as
  `@parallel_indices (i, j)` translate to `i, j = @index(Global, NTuple)`.
- KA kernels must not contain early `return` statements (the final `return nothing` must be
  dropped; `continue` inside loops is fine, as are `if/else` branches).
- Add `@Const` annotations to read-only array arguments (`V`, `grid_vi`, …) where possible — free
  aliasing information for the GPU compilers.

### 2.2 Kernel launch

Before (`src/Particles/Advection/advection.jl:45`):

```julia
@parallel (@idx ni) advection_kernel!(coords, method, V, index, grid_vi, local_limits, dxi, dt)
```

After — via one small internal helper so call sites stay one-liners:

```julia
# src/launch.jl
@inline function launch!(backend, kernel::F, ni::NTuple{N, Int}, args::Vararg{Any, NA}) where {F, N, NA}
    kernel(backend)(args...; ndrange = ni)
    KernelAbstractions.synchronize(backend)   # PS @parallel is synchronous; keep semantics (see §3.2)
    return nothing
end

# call site
launch!(ka_backend(particles), advection_kernel!, ni, coords, method, V, index, grid_vi, local_limits, dxi, dt)
```

where `ka_backend(particles) = KernelAbstractions.get_backend(particles.index.data)` is a new
one-line accessor (add equivalents for `MarkerChain`, `PassiveMarkers`, `PhaseRatios`, and a plain
`AbstractArray` fallback for the semilagrangian routines that only receive grid arrays).

- Launch sites currently passing ranges instead of sizes (`@parallel (1:np)`,
  `@parallel (1:nx, 1:ny)` — e.g. `src/PassiveMarkers/advection.jl:12`, `src/MarkerChain/areas.jl`)
  pass the lengths as `ndrange`.
- The **offset range** `@parallel (2:(n - 1))` in `src/MarkerChain/bilinear_MC.jl:129` needs
  `ndrange = n - 2` plus `i = @index(Global) + 1` inside the kernel (KA ndranges are 1-based).
- JustPIC's own `@idx` macro (`src/Utils.jl:23`) becomes unnecessary at kernel-launch sites; keep it
  exported during the transition, delete once no user-facing code depends on it.
- Workgroup size: start with the KA default (`nothing` → heuristic). If profiling warrants it, use
  `kernel(backend, 64)` for 1D and `kernel(backend, (32, 8))` for 2D/3D launches — expose as a
  package-internal constant, not user API.

### 2.3 CellArray allocation (`@fill`)

PS's `@fill(NaN, ni..., celldims = (max_xcell,))` allocates on whatever backend the enclosing module
was initialized with. Replace with an explicit-backend helper next to the existing `CA`:

```julia
function cell_array(x::T, backend, ni::NTuple{N, Int}; celldims = (1,), eltype = T) where {T, N}
    A = CA(backend, ni; eltype = SVector{prod(celldims), eltype})  # or direct CellArray ctor
    fill!(A.data, x)
    return A
end
```

All `@fill` call sites already pass `celldims`/`eltype` explicitly, so nothing relies on the
`Float64` default baked into `@init_parallel_stencil` — the mechanical rewrite is safe. The `init_particles` /
`init_markerchain` / `SubgridDiffusionCellArrays` functions gain the backend from their existing
`::Type{<:AbstractBackend}` first argument (map `CPUBackend → CPU()`, `CUDABackend → CUDABackend()`,
`AMDGPUBackend → ROCBackend()` with a small `ka_backend(::Type{...})` function).

### 2.4 Atomics

Replace the three per-module `@myatomic` definitions with `KernelAbstractions.@atomic` (which
lowers to Atomix and supports CPU, CUDA, AMDGPU). The only user is
`src/PassiveMarkers/particle_to_grid.jl:47-48`. Drop the `Atomix` direct dependency if nothing else
uses it after this.

---

## 3. Known gotchas (read before starting)

### 3.1 ⚠ `@index` name collision — the single biggest hazard (findings verified empirically)

JustPIC uses **CellArraysIndexing's `@index`** pervasively (**170 sites across 27 files**) for
particle-slot *element* access (`@index p[k][ipart, I...]`, `@index p[k][ipart, I...] = v`). KA's
`@kernel` needs **its own** `@index(Global, …)` to get the work-item index. Both macros are named
`@index`; they cannot both be the bare `@index` binding in one module.

Two facts settled by reading the installed sources and running probe kernels — **both correct my
earlier draft**:

- **`@cell` is NOT a drop-in for `@index`.** They lower to *different* functions
  (`CellArraysIndexing/src/macros.jl`): `@index x[i, I...]` → `getcellindex(x, i, I...)` (one scalar
  *inside* a cell), while `@cell x[I...]` → `getcell(x, I...)` (the *whole* cell SVector/SMatrix,
  no per-element index). Swapping `@index`→`@cell` would change semantics. Draft Mitigation #1 was
  wrong — discard it.
- **KA's `@index` must be bare.** `@kernel`'s CPU codegen syntactically finds `I = @index(...)`
  assignments and splices the work-item index into the call (`emit_cpu`: `push!(rhs.args, idx)`);
  the CPU `__index_*` methods require that extra `idx::CartesianIndex` (`cpu.jl`). Verified: a
  bare `@index(Global, Linear)` kernel runs on CPU; the same kernel with `KernelAbstractions.@index`
  qualified throws `MethodError: no method matching __index_Global_Linear(::CompilerMetadata…)` on
  CPU. (Qualified would "work" on GPU, making this an asymmetric trap.) So KA's `@index` stays bare
  and in `I = @index(Global, NTuple)` assignment form.

**Chosen strategy (the only consistent one):**
- In kernel-defining modules (`_2D`/`_3D`, and later the two extensions): `using KernelAbstractions`
  so **bare `@index` = KA's**, and `import CellArraysIndexing as CAI` so **CellArraysIndexing's
  element access is written `CAI.@index`** at all 170 sites. Drop `CellArraysIndexing` from the
  `using` list (keep the non-`@index` names it needs — `@cell`, `getcell`, … — via a selective
  `using CellArraysIndexing: …` if used bare).
- The **top-level `JustPIC` module** defines no kernels, so it keeps `using CellArraysIndexing` with
  bare `@index` = CellArraysIndexing's and continues to `export @index` — the **public API is
  unchanged** (`using JustPIC; @index p[...]` still works).
- Consequence: `JustPIC._2D.@index` / `JustPIC._3D.@index` now resolve to KA's, so internal tests
  that wrote `JustPIC._2D.@index CA[...]` (e.g. `test/test_CellArrays.jl`) must switch to
  `CellArraysIndexing.@index` (or `getcellindex`/`setcellindex!`). These are our own tests.

**Sequencing that keeps CI green throughout** (important — a module can't be half-migrated, since
adding `using KernelAbstractions` makes every remaining bare `@index` ambiguous at once):
1. **Qualification pass (safe, self-contained, still on ParallelStencil):** replace all 170 bare
   CellArraysIndexing `@index` → `CAI.@index`, add `import CellArraysIndexing as CAI`. Behaviour is
   identical (bare `@index` still resolves to CellArraysIndexing at this point). Land + test green.
2. **Introduce KA (`using KernelAbstractions`) and port kernels/launches file-by-file.** Now bare
   `@index` is free to become KA's with no conflict, because no file uses bare `@index` anymore.

Add a CI lint that forbids bare `@index(...)`-that-isn't-`@index(Global/Local/Group`-outside a
`CAI.`/`CellArraysIndexing.` qualifier in kernel modules, to prevent regressions.

> Note on the author's suggestion (`CAI.@index` + `KA.@index`): the `CAI.@index` half is exactly
> right and adopted. The `KA.@index` half does **not** work — KA's index macro must stay bare (per
> the verified CPU `MethodError` above).

### 3.2 Synchronization semantics

PS's `@parallel` blocks until the kernel completes; KA launches are **asynchronous**. Several hot
paths assume completion ordering:
- `advection!` → `move_particles!` → `inject_particles!` sequences (device-side ordering is fine on
  a single stream, but host reads and MPI are not),
- `update_cell_halo!` (ImplicitGlobalGrid/MPI must not read half-written buffers),
- reductions/host logic in `move_particles!` and injection that inspect kernel results.

Phase 1 keeps a `synchronize` inside `launch!` (strictly PS-equivalent, zero risk). A later
optimization pass may remove per-launch syncs and synchronize only before host access / MPI — do
this as its own PR with benchmarks, never mixed into the mechanical port.

### 3.3 `CuCellArray` / `ROCCellArray` come from PS today

`ext/JustPICCUDAExt.jl:172` calls `CuCellArray{eltype}(undef, dims)` — that alias is *defined by*
`@init_parallel_stencil(CUDA, …)`. After removing PS, call `CellArrays.@define_CuCellArray()`
(resp. `@define_ROCCellArray()`) once at the top of each extension. Note the extensions also define
their own `CuCellArray(::Type{T}, undef, dims)` *function* overloads (`ext/JustPICCUDAExt.jl:8`) —
keep those, they're independent.

### 3.4 CPU performance risk

PS's `Threads` backend generates tight nested `@threads` loops; KA's `CPU()` backend has historically
been slower for small kernels (task-based, dynamic ndrange). Mitigations:
- benchmark before/after with `scripts/temperature_advection_timer.jl` (2D) and a 3D equivalent,
  plus the test-suite runtime as a proxy;
- try `CPU(; static = true)` for static scheduling;
- acceptance gate: ≤ 10% regression on CPU advection benchmarks, none on GPU.

### 3.5 Kernel argument `isbits` requirements

KA (like PS) requires GPU kernel args to be isbits. Everything currently passed already crosses a
PS kernel boundary on GPU, so no new violations are expected — but `LinRange`/`StepRangeLen` grid
tuples and the `di::NamedTuple` of vectors should be spot-checked on CUDA early (Phase 3 smoke
test) rather than at the end.

---

## 4. Target architecture

```
src/JustPIC.jl            # structs, backend tags, ka_backend() mapping, TA
src/common.jl             # included ONCE per dimension module (Phase 1) → ONCE total (Phase 5)
src/JustPIC_CPU.jl        # _2D/_3D modules WITHOUT @init_parallel_stencil; CPU CA()
ext/JustPICCUDAExt.jl     # ~150 lines: @define_CuCellArray, TA/CA methods, host↔device conversions
ext/JustPICAMDGPUExt.jl   # ~150 lines: same for ROC
```

Key change in dispatch philosophy: instead of *compiling the same generic code once per backend
module*, kernels are generic and the backend is a **runtime value** derived from the arrays
(`get_backend`). The entire forwarding layer in the extensions
(every `JustPIC._2D.f(::Particles{CUDABackend}, …) = f(…)` method) is deleted — the generic methods
in `common.jl` already handle any array type once launches go through `launch!(backend, …)`.

The `_2D`/`_3D` split remains **only** as an API namespace (users do `using JustPIC._2D`). In
Phase 1 both modules still `include("common.jl")` (now backend-generic, so 2 includes instead
of 6). Phase 5 (optional) collapses to a single include with `_2D`/`_3D` as re-export shims —
possible because nearly all kernels are already dimension-agnostic (`(I...)` + `NTuple{N}`
dispatch); the handful of dimension-specific ones (MarkerChain is inherently 2D) dispatch on tuple
arity anyway.

---

## 5. Phased execution plan

Each phase is one reviewable PR; the package must be green on CPU CI after every phase, and on GPU
CI (Buildkite) after Phases 3–4.

### Phase 0 — scaffolding (small PR)
- [x] Add `KernelAbstractions` to `[deps]` (compat `"0.9"`); keep PS temporarily (both coexist).
- [x] Add `ka_backend(::Type{<:AbstractBackend})` mapping and `backend(::AbstractParticles)`
      accessors in `src/JustPIC.jl` / `src/particles.jl`.
      Done as `ka_backend(...)` accessors in `src/launch.jl` for arrays, `Particles`,
      `MarkerChain`, `PassiveMarkers`, and `PhaseRatios`; there is not a separate `backend(...)`
      alias.
- [x] Add `launch!` helper and CellArray allocation helper.
      Done as `launch!` plus `cell_array` / `undef_cell_array` in `src/launch.jl`. Some older
      `@fill`-based allocation paths still need to be switched over in Phase 1.
- [ ] Benchmark baseline: record `scripts/temperature_advection_timer.jl` numbers (CPU 1/4/8
      threads, CUDA) on `main`, commit results to the PR description.

### Phase 1 — port `src/` kernels, CPU only (the big mechanical PR)
Convert per feature area, in dependency order; each bullet = one commit:
- [x] `src/Interpolations/` (4 kernel files; `grid_to_particle.jl`, `particle_to_grid.jl`,
      `particle_to_grid_centroid.jl`, `centroid_to_particle.jl`)
- [x] `src/Particles/Advection/` (6 files: advection{,_LinP,_MQS}, backtracking{,_LinP,_MQS})
- [x] `src/Particles/` (move_safe, injection, forced_injection, particles_utils [`@fill`], utils;
      note `move.jl` is currently not included by `common.jl` — port or delete it, don't leave it stale)
      Done: `move.jl` was ported rather than deleted.
- [x] `src/PhaseRatios/` (centers, vertices, midpoints)
- [x] `src/MarkerChain/` (init, move, resample, areas, bilinear_MC [offset range §2.2],
      Advection/* — 9 files)
- [x] `src/PassiveMarkers/` (init, advection, grid_to_particle, particle_to_grid [`@atomic`])
      Done: `particle_to_grid.jl` now uses `KernelAbstractions.@atomic`.
- [x] `src/Physics/subgrid_diffusion.jl`, `src/CellArrays/ImplicitGlobalGrid.jl`
- [x] In every kernel body: qualify CellArraysIndexing element access as `CAI.@index`, drop
      `return nothing`, hoist work-item indices via `@index(Global, NTuple)`.
- [x] `src/JustPIC_CPU.jl`: delete `@init_parallel_stencil` and `@myatomic`; keep module structure.
- [x] Full CPU test suite green via `Pkg.test(; test_args=["--backend=CPU"])`.
- [x] MPI script `scripts/temperature_advection_MPI_ci.jl` if runnable locally.
      Done as a one-rank CPU smoke with `JUSTPIC_MPI_CI_N=16`, `JUSTPIC_MPI_CI_NITER=1`.

### Phase 2 — remove PS from CPU path
- [x] Delete `using ParallelStencil` from `src/`; CPU package no longer loads PS.
- [x] `test/test_2D.jl` / `test_3D.jl`: remove `@init_parallel_stencil` blocks and
      `using ParallelStencil` (keep the backend-selection `ENV` logic).
- [ ] CPU benchmarks vs Phase 0 baseline; investigate/fix regressions > 10% (§3.4).
      Partial: local post-port CPU timer numbers were collected, but no pre-migration Phase 0
      baseline exists in this worktree for a regression comparison.

### Phase 3 — rewrite CUDA extension
- [x] Remove active `ParallelStencil` import/init, `@myatomic`, and extension-local `@parallel`
      launches from `ext/JustPICCUDAExt.jl`; add KA imports and `undef_cell_array` for CUDA.
- [ ] Strip `ext/JustPICCUDAExt.jl` `_2D`/`_3D` submodules: delete `@init_parallel_stencil`,
      the `include("../src/common.jl")` re-includes, `@myatomic`, and the entire forwarding-method
      layer.
- [ ] Keep/move to top level: `@define_CuCellArray()` (§3.3), `TA(::Type{CUDABackend}) = CuArray`,
      `CA(::Type{CUDABackend}, …)`, all `CUDA.CuArray(::Particles/MarkerChain/PhaseRatios/CellArray)`
      conversions, the `Particles(coords, index::…CuArray…)` reconstruction constructors.
- [ ] `src/CUDAExt/CellArrays.jl`: fold into the extension or port as-is.
- [ ] Smoke test early (§3.5), then full Buildkite CUDA suite
      (`Pkg.test(test_args=["--backend=CUDA"])`).
- [ ] GPU benchmark vs baseline.

### Phase 4 — rewrite AMDGPU extension
- [x] Remove active `ParallelStencil` import/init, `@myatomic`, and extension-local `@parallel`
      launches from `ext/JustPICAMDGPUExt.jl`; add KA imports and `undef_cell_array` for AMDGPU.
- [ ] Collapse `ext/JustPICAMDGPUExt.jl` forwarding layer after GPU runtime validation.
- [ ] Buildkite AMDGPU suite + CSCS CI (`ci/cscs-mi300.yml`, `ci/cscs-gh200.yml`).

### Phase 5 — cleanup & consolidation
- [x] Remove `ParallelStencil` from `test/Project.toml`.
- [x] Remove `ParallelStencil` from `Project.toml` `[deps]`/`[compat]`.
- [x] Remove direct `Atomix` dependency from `Project.toml`.
- [x] Update `scripts/*.jl` and `docs/` examples (they call `@init_parallel_stencil` and
      `using ParallelStencil`); update `docs/src/mixed_CPU_GPU.md` and README if they mention PS.
- [ ] Optional: collapse `common.jl` to a single include with `_2D`/`_3D` re-export shims.
- [ ] Optional follow-ups (separate PRs): per-launch sync elimination (§3.2), Metal backend,
      `@Const` audit.
- [x] Add the "no bare `@index` inside `@kernel`" lint check to CI.
      Done as the `Migration lint` test in `test/test_Aqua.jl`; it scans `src/`, `ext/`,
      `scripts/`, and `test/`, and also rejects stale active ParallelStencil/Atomix
      macro/import usage.
- [ ] Bump minor version (0.7.0) — internals overhaul, API unchanged; note in release notes that
      users no longer need `@init_parallel_stencil` boilerplate in their own scripts.

---

## 6. Validation strategy

1. **Unit tests** per backend at every phase (custom runner: `test/runtests.jl`, `--backend=` flag).
2. **Physical regression tests** — run before/after and diff results quantitatively:
   `scripts/Zalesack_disk.jl`, `scripts/donut_advection.jl`, `scripts/pureshear_ALE.jl` (+ restart),
   `scripts/rotating_marker_chain.jl` (MarkerChain path), 2D/3D MPI advection scripts.
   Bitwise identity is *not* expected (different launch order ⇒ different atomic-add order in
   passive-marker p2g; different FP contraction), so compare against analytic solutions / norms with
   tolerances, same as the existing tests do.
3. **Performance gates** (§3.4): timer script on CPU(1/4/8 threads), CUDA, AMDGPU; ≤ 10% CPU
   regression, ~0% GPU regression accepted.
4. **Downstream check**: JustRelax.jl is the main consumer (Downstream.yml CI) — run its test suite
   against the branch before merging Phase 5, since it currently co-initializes PS in the same
   session (verify no load-order/versioning conflicts while both packages transition).

---

## 7. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `@index` collision silently corrupts kernels | certain if unhandled | data corruption | §3.1: qualify CellArraysIndexing calls as `CAI.@index` + CI lint; caught immediately by advection tests otherwise |
| CPU perf regression (KA CPU backend) | medium | perf | §3.4 benchmarks, static scheduling, acceptable-loss gate |
| Missing sync before MPI/host read | medium | wrong results, flaky MPI | conservative per-launch sync in Phase 1; relax later in isolation |
| Non-isbits kernel arg on GPU | low | GPU compile error (loud) | early Phase 3 smoke test |
| Downstream (JustRelax) breakage | medium | ecosystem | Downstream CI on the branch; coordinate release |
| AMDGPU-specific atomics/KA quirks | low-medium | AMDGPU CI red | Phase 4 isolated; Buildkite + CSCS coverage exists |

---

## 8. Effort estimate

| Phase | Size |
|---|---|
| 0 scaffolding | ~½ day |
| 1 core kernel port (~60 kernels, ~50 launches, 33 files) | 3–5 days (mechanical but must be careful with §3.1) |
| 2 PS removal from CPU + benchmarks | ~½ day |
| 3 CUDA ext rewrite | 1–2 days (mostly deletion) + GPU CI iteration |
| 4 AMDGPU ext rewrite | ~1 day (mirror of 3) |
| 5 cleanup, scripts/docs, release | 1–2 days |

Total: roughly **1.5–2 focused weeks**, dominated by Phase 1 and GPU CI turnaround.

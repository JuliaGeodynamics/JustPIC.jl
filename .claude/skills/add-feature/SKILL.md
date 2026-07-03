---
name: add-feature
description: Checklist for adding new functionality to JustPIC
---

# Add Feature

Follow this checklist when adding new functionality (interpolation scheme, advection
variant, particle operation, …).

## Checklist

1. **Implement in the right feature area** under `src/` (`Particles/`, `Interpolations/`,
   `MarkerChain/`, `PassiveMarkers/`, `PhaseRatios/`, `Physics/`, `IO/`)
2. **Write kernels dimension-generically** where possible (one `@kernel` covering 2D and
   3D via `N`-tuples), following `.claude/rules/kernel-rules.md`:
   - `@kernel` + KA `@index`; `CAI.@index` for cell-element access
   - type-stable, allocation-free, `@inline` helpers
   - launch via `launch!(ka_backend(x), kernel!, ndrange, args...)`
   - eltype-generic numerics (Metal has no Float64)
3. **Register in `src/common.jl`**: add the `include` in the matching section and the
   `export` right below it — this is the single wiring point for all backends
4. **Don't touch `ext/`** unless the feature needs new allocation or host↔device
   conversion primitives; algorithms never go there
5. **Add tests** to `test_2D.jl` *and* `test_3D.jl` (testset names suffixed "2D"/"3D"),
   using the file's `backend` constant so they run on CPU and GPU unchanged. If a new
   `test_*.jl` file is warranted, register it in `test/runtests.jl` (CPU include list +
   `gpu_testfiles`)
6. **Prefer analytic assertions**: exact reproduction of linear fields, conserved particle
   counts, phase ratios summing to 1 (see `.agents/validation.md`)
7. **Update `docs/src/API.md`** and the relevant docs page if the public API changed
8. **Verify**: single-file CPU run first (`/run-tests`), then the full CPU suite, then
   `--backend=CUDA` if a GPU is available
9. **Format**: `git runic main` before pushing

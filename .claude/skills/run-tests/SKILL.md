---
name: run-tests
description: Run targeted JustPIC tests, prioritized by what's likely to break
---

# Run Tests

Run targeted tests first, prioritized by what's most likely to fail given recent changes.
Only fall back to the full suite once the targeted files pass.

## Step 1: Identify What Changed

Map the changed area to test files (testsets for a feature usually exist in both
`test_2D.jl` and `test_3D.jl`):

| Changed area | Test files to run first |
|---|---|
| `src/Particles/` (move/injection/utils) | `test_2D.jl`, `test_3D.jl` (Particles initialization, Cell index, Forced injection, Miniapps testsets) |
| `src/Particles/Advection/`, `src/Advection/` | `test_integrators.jl`, then advection testsets in `test_2D.jl`/`test_3D.jl` |
| `src/Interpolations/` | `test_interpolation_kernels.jl`, Interpolations testsets in `test_2D.jl`/`test_3D.jl` |
| `src/CellArrays/`, `src/launch.jl` | `test_CellArrays.jl` |
| `src/MarkerChain/` | `test_save_load.jl` (only existing coverage — consider adding testsets to `test_2D.jl` when touching this area) |
| `src/PassiveMarkers/` | Passive markers testsets in `test_2D.jl`/`test_3D.jl` |
| `src/PhaseRatios/` | Phase ratios testsets in `test_CellArrays.jl` |
| `src/Physics/` | Subgrid diffusion testsets in `test_2D.jl`/`test_3D.jl` |
| `src/IO/` | `test_save_load.jl` |
| Exports / `Project.toml` / new deps | `test_Aqua.jl` |
| `ext/` | full GPU run (`--backend=CUDA`) — extensions are only exercised there |

## Step 2: Run the Most Likely File First

Single file, CPU (fast iteration; needs TestEnv in your default env once:
`julia -e 'using Pkg; Pkg.add("TestEnv")'`):

```sh
julia --project=. -e 'using TestEnv; TestEnv.activate();
                      ENV["JULIA_JUSTPIC_BACKEND"] = "CPU";
                      include("test/test_2D.jl")'
```

Swap the env var to `"CUDA"` / `"AMDGPU"` to run the same file on a GPU.

## Step 3: Fix and Iterate

1. Fix the failure, re-run the same file to confirm
2. Move to the next most likely file
3. Run `test_Aqua.jl` for any change touching exports, deps, or method signatures
   (ambiguity checks)

## Step 4: Full Suite Before Handing Off

```sh
julia --project=. -e 'using Pkg; Pkg.test()'                                   # CPU
julia --project=. -e 'using Pkg; Pkg.test(; test_args=["--backend=CUDA"])'     # GPU
```

## Notes

- New `test_*.jl` files must be registered in `test/runtests.jl` (both the CPU include
  list and the `gpu_testfiles` tuple) or they never run
- GPU-only failures are usually type instability — reproduce on CPU first
- The GPU path spawns a subprocess per file with `--project=<Pkg.test sandbox>`; that flag
  is load-bearing, don't "simplify" it away

---
paths:
  - test/**/*.jl
---

# Testing Rules

## Running

```sh
# Full suite, CPU
julia --project=. -e 'using Pkg; Pkg.test()'

# Full suite, GPU backend
julia --project=. -e 'using Pkg; Pkg.test(; test_args=["--backend=CUDA"])'   # or AMDGPU

# Single file, fast iteration (TestEnv in default env: Pkg.add("TestEnv") once)
julia --project=. -e 'using TestEnv; TestEnv.activate();
                      ENV["JULIA_JUSTPIC_BACKEND"] = "CPU";
                      include("test/test_2D.jl")'
```

GPU test deps come from `[extras]`/`[targets]` in the root `Project.toml` (no
`test/Project.toml`). Each test file reads `ENV["JULIA_JUSTPIC_BACKEND"]` to pick the
backend.

## The custom runner

`test/runtests.jl` keeps **two explicit file lists**:

- CPU path: in-process `include`s (`test_Aqua.jl`, `test_2D.jl`, `test_integrators.jl`,
  `test_CellArrays.jl`, `test_save_load.jl`, `test_3D.jl`)
- GPU path: the `gpu_testfiles` tuple, one subprocess per file, spawned with
  `--project=<Pkg.test sandbox>` — that flag is load-bearing, don't remove it

**A new `test_*.jl` file must be added to these lists or it will never run.** Check for
this whenever creating a test file.

## Writing tests

- Use the `backend` constant from the top of the file so the same test runs on CPU and GPU
- No scalar indexing of device arrays in new tests — `Array(...)` the data first; don't
  extend the legacy `allowscalar(true)`
- Prefer analytic checks: known trajectories, exact linear-field reproduction, phase
  ratios summing to 1, conserved particle counts (`sum(p.index.data)`)
- Minimal grid sizes — CI runs this on 6+ platform/version combinations
- 2D variants go in `test_2D.jl`, 3D in `test_3D.jl`; keep testset names suffixed
  "2D"/"3D" like the existing ones

## Debugging

- GPU-only failure ⇒ suspect type instability: reproduce on CPU first
- Manifest/version weirdness: delete `Manifest.toml`, re-`instantiate`

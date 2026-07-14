# Testing Guidelines

## Running Tests

```sh
# Full suite, CPU backend (default)
julia --project=. -e 'using Pkg; Pkg.test()'

# Full suite against a GPU backend
julia --project=. -e 'using Pkg; Pkg.test(; test_args=["--backend=CUDA"])'    # or AMDGPU
```

GPU test dependencies (CUDA/AMDGPU/Metal) come from `[extras]`/`[targets]` in the root
`Project.toml` — `Pkg.test` puts them in its temporary sandbox environment. There is no
`test/Project.toml`.

### Fast iteration on a single test file

Use TestEnv.jl (one-time `julia -e 'using Pkg; Pkg.add("TestEnv")'` in your default env):

```sh
julia --project=. -e 'using TestEnv; TestEnv.activate();
                      ENV["JULIA_JUSTPIC_BACKEND"] = "CPU";   # or "CUDA" / "AMDGPU"
                      include("test/test_2D.jl")'
```

Every test file reads `ENV["JULIA_JUSTPIC_BACKEND"]` at the top to pick the backend
(`CPU`, `CUDA.CUDABackend`, `AMDGPU.ROCBackend`).

## How the custom runner works

`test/runtests.jl` is a custom runner, not a plain `@testset` tree:

- It parses `--backend=` from `test_args` (default `CPU`) and sets
  `ENV["JULIA_JUSTPIC_BACKEND"]`.
- **CPU path**: `include`s an explicit list in-process — `test_Aqua.jl`, `test_2D.jl`,
  `test_integrators.jl`, `test_CellArrays.jl`, `test_save_load.jl`, then `test_3D.jl`.
- **GPU path**: spawns one fresh `julia` subprocess per file in the `gpu_testfiles` tuple
  (`test_2D.jl`, `test_3D.jl`, `test_CellArrays.jl`, `test_interpolation_kernels.jl`,
  `test_save_load.jl`), passing `--project=<Pkg.test sandbox>` and a `JULIA_LOAD_PATH`
  that includes the repo root. The `--project` flag pointing at the sandbox is
  load-bearing — without it the subprocess cannot find CUDA/AMDGPU.

**A new test file is not picked up automatically.** Add it to the CPU `include` list *and*
to `gpu_testfiles` (if it should run on GPU). Orphaned test files are a recurring mistake.

## Writing Tests

- Name files `test_<area>.jl`; put 2D and 3D variants of a feature in `test_2D.jl` /
  `test_3D.jl` rather than new files when they fit
- Get the backend from the `backend` constant defined at the top of each test file so the
  test runs on CPU and GPU unchanged
- Avoid scalar indexing of device arrays in new tests — move data to host with `Array(...)`
  first (the existing `allowscalar(true)` at file top is legacy, don't extend it)
- Test numerical accuracy where analytical solutions exist (advection in analytic velocity
  fields, sum of phase ratios == 1, particle counts conserved)
- Use minimal grid sizes to keep CI time down
- `test_Aqua.jl` runs Aqua.jl package hygiene plus a migration lint testset; two
  `_grid2particle` ambiguities are deliberately excluded there

## Debugging

- GPU-only failures ("dynamic function invocation", wrong results) are usually type
  instability: reproduce on CPU first, then check inference on the kernel
- Julia version/manifest weirdness: delete `Manifest.toml`, `Pkg.instantiate()` again
- Occupancy/injection bugs: assert `sum(p.index.data)` before/after `move_particles!` and
  `inject_particles!` — counts must match expectations cell-by-cell

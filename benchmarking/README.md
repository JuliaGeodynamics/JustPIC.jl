# JustPIC performance benchmarks

This environment measures steady-state JustPIC performance independently from the
correctness test suite. Every timed case receives fresh input state, runs one evaluation per
sample, and validates its numerical invariants outside the timed region.

Run the initial CPU suite from the repository root:

```sh
julia --project=benchmarking benchmarking/setup.jl
julia --project=benchmarking benchmarking/run_benchmarks.jl
```

The setup step develops the repository checkout into the isolated benchmark environment.
Run it again after deleting or regenerating `benchmarking/Manifest.toml` or switching Julia
versions; otherwise Julia 1.10 may resolve the registered JustPIC release instead of the
checkout being measured.

The runner writes `benchmark_results.json`. Each entry contains the fields required by
`github-action-benchmark` (`name`, `unit`, and `value`) together with timing dispersion,
allocations, throughput, problem parameters, package versions, hardware metadata, the git
revision, and whether the worktree was dirty.

## Performance model

Wall time and allocations are measured. FLOPs and memory traffic use a versioned algorithmic
model because portable Julia APIs cannot read equivalent hardware counters on every CPU and
GPU. The result labels these fields with `performance_metric_source = "algorithmic_model"`.
An FMA counts as two FLOPs; add, subtract, multiply, divide, square root, and reciprocal each
count as one. Integer arithmetic, comparisons, indexing, and control flow are excluded.
Modeled bytes are logical scalar reads and writes, not cache- or DRAM-counter measurements.

For grid size `n`, let `Np = 4n²` particles, `Ns = 8n²` particle slots,
`Nv = (n + 1)²` surface or grid vertices, and `Nc = n²` cells. The `justpic_cpu_v1` model is:

| Benchmark | Modeled FLOPs | Modeled bytes |
| --- | ---: | ---: |
| Particle workflow | `54Np` | `104Np + 2Ns` |
| Interpolation round-trip | `54Np + Nv` | `76Np + 4Nv + 40Nc` |
| MarkerSurface update | `317Nv + 14Nc` | `294Nv + 37Nc` |

The byte counts hold for `Float32` fields and scale with the element size, so a `Float64` run
models twice the traffic. The `2Ns` particle-index bytes are the one exception: they are
`Bool` slots and stay fixed. FLOP counts are independent of the element type.

Each JSON record reports `arithmetic_intensity_flops_per_byte`,
`effective_flops_per_second`, `effective_gflops_per_second`, and
`effective_bandwidth_gb_per_second`. These are effective application metrics derived from the
measured median runtime and the model above. Hardware-counter measurements should use a
distinct source label rather than silently replacing this model.

Useful options:

```sh
julia --project=benchmarking benchmarking/run_benchmarks.jl --samples=20
julia --project=benchmarking benchmarking/run_benchmarks.jl --group=MarkerSurface
julia --project=benchmarking benchmarking/run_benchmarks.jl --output=out/results.json
julia --project=benchmarking benchmarking/run_benchmarks.jl --backend=CUDA
julia --project=benchmarking benchmarking/run_benchmarks.jl --precision=Float32
```

`--backend` accepts `CPU` (default), `CUDA`, `AMDGPU`, or `Metal`. GPU cases use the same
problem sizes as the CPU, synchronize the device before each timing ends, and record the
device name in `metadata.device` and `metadata.hardware_fingerprint`.

`--precision` accepts `Float64` (default) or `Float32` and sets the element type of every
field, grid, and velocity array, as well as of the STREAM triad and FMA-chain probes that
measure the roofline ceilings. Metal has no `Float64` and runs `Float32` unless `--precision`
says otherwise. The element type is part of each benchmark name (`..._F64`, `..._F32`) and is
recorded in `metadata.float_type`, so the two precisions form separate dashboard series.

## Comparing against a base revision

```sh
julia --project=benchmarking benchmarking/compare.jl --rev=main
julia --project=benchmarking benchmarking/compare.jl --rev=origin/main --backend=Metal --group=MarkerSurface
```

The script checks out `--rev` (default `main`) in a temporary git worktree, installs the
working tree's benchmark harness into it, and runs the suite on that revision and then on the
working tree, on the same machine and with the same arguments. It prints each benchmark's
median time, interquartile spread, candidate-to-baseline ratio, and allocations. Both runs
must share a backend, element type, and hardware fingerprint. A ratio within the printed
spread is not evidence of a change; repeat the comparison before acting on it.

## Performance dashboard

The dashboard is the **Performance** page of the documentation (`docs/src/performance.md`,
rendered by `docs/src/components/PerformanceDashboard.vue`). The page reads
`benchmark_history.json` from the `benchmark-data` branch when it is viewed, so new results
appear without rebuilding the documentation. The history file is aggregated from per-commit
result files:

```sh
julia --project=benchmarking benchmarking/build_dashboard.jl \
    --input=out/benchmark_results.json \
    --output=out/benchmark_history.json
```

Repeat `--input=PATH` for historical result files. Inputs must identify unique commit,
backend, and hardware combinations; the builder fails instead of silently selecting between
duplicate runs.

## Continuous tracking

Buildkite runs the suite in `.buildkite/run_tests.yml` on the `cuda`, `rocm`, and `metal`
queues (the CUDA and Metal agents also run the CPU backend) and uploads each
`benchmark_results_<label>.json` as a build artifact. When Buildkite reports the final
status of a `main` commit, `.github/workflows/PublishBenchmarks.yml`:

1. downloads that build's benchmark artifacts, which are public on JuliaGPU's Buildkite, and
   rejects any measured on a dirty worktree;
2. stores them as `results/<commit>/<label>.json` on the `benchmark-data` branch;
3. regenerates `benchmark_history.json` on that branch from every stored result.

The Performance page fetches that file from GitHub, so it shows new results within minutes
of the push without a documentation rebuild.

Per-commit result files are the source of truth; `benchmark_history.json` and the page are
derived from them. The workflow uses only the repository's built-in `GITHUB_TOKEN`.

The publish workflow requires the `benchmark-data` branch; to create it:

```sh
git switch --orphan benchmark-data
echo '{"schema_version": 1, "repository_url": "https://github.com/JuliaGeodynamics/JustPIC.jl", "runs": []}' \
    > benchmark_history.json
git add -f benchmark_history.json
git commit -m "Initialize benchmark history"
git push origin benchmark-data
```

Shared CI agents are not dedicated benchmark machines, and GPU models can vary between
builds of the same queue. Compare results only within one hardware fingerprint, and derive
any regression threshold from repeated measurements on that fingerprint rather than from an
arbitrary percentage.

The initial groups are:

- `Particle workflow`: one periodic RK2 advection and movement step;
- `Interpolation`: particle-to-grid followed by grid-to-particle interpolation;
- `MarkerSurface`: a complete surface update with interpolation, advection, and smoothing.

MPI and checkpoint benchmarks, automated PR comparisons, and regression thresholds are not
yet covered.

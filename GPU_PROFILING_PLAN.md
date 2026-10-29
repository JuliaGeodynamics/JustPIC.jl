# GPU profiling plan: `advection_MQS!` and `move_particles!`

## Current checkpoint (2026-10-29)

F1-F7 are implemented on `adm/gpu-perf` (`59b0276`; plan update
`ac50925`). F9 (cached classification bounds) and F8 (packed counters)
are implemented in the working tree on top of `ac50925` and pass the
CPU/CUDA movement gates; see "Status after F9 and F8" below for the
next experiment. The RTX 3080 measurements below are historical; they are
not a baseline for the current Windows host, which has an RTX 5070 Ti (16 GB).
The A/B against `main` on this host is in `GPU_PERFORMANCE_REPORT.typ`.

The old session-local driver is missing. `nsys` and `ncu` are not on PATH
on this host. Restore a driver under `scratchpad/prof/` (keep it local, as
originally intended), use CUDA event timing and `CUDA.@profile` first,
and record tool availability rather than assuming the old counter-permission
failure still applies. Do not change package dependencies for profiling.

### Next work and acceptance

1. Restore a headless, parameterized driver with NVTX ranges. Create one
   seeded particle snapshot per case and restore it outside every timed
   sample; advection and movement mutate their inputs. Warm each operation
   and specialization before collecting at least 10 samples. Record median
   wall time and GPU event time separately. Record device, Julia/package
   versions, revision, dimensions, precision, occupancy and dropped count.
2. Rebaseline F7 on this device: start with 2D n=256 and 3D n=64 in both
   precisions; compare pull with colored sweeps on identical advected
   snapshots. Keep no-injection single-call measurements separate from
   multi-step advection/move/injection runs, since full cells change the
   number of pull rounds and the drop count.
3. Profile classification, pull and clear, including round count, host
   reductions, allocations and launch count. Reassess M1 on this GPU;
   historical F3 disproved it as the main RTX 3080 bottleneck. F7 already
   tried thread-per-slot classification without a gain.
4. Select one further experiment from the measured dominant cost. Preserve
   the periodic/long-jump fallback, settled-full-cell and cycle handling,
   coordinate/field pairing, and occupancy. Compare survivors by particle
   identity and fields, not just counts; compare slot sets where ordering
   is unspecified. Reject regressions or gains within measurement noise.
5. Before retaining a source change, run the relevant 2D/3D movement tests
   on CPU and CUDA, with Float32 and Float64; format Julia changes with
   Runic. Record exact commands and results here. AMDGPU/Metal results
   require those devices and must not be inferred from CUDA success.

Continue from measured bottlenecks. Keep each production change separate
from local prototypes and record its performance and correctness gates.

### Continuation results: RTX 5070 Ti

Completed the initial four-case baseline and the F8 counter experiment;
F9 below is the subsequent production optimization. Local driver/prototype files:
`scratchpad/prof/driver.jl` and `scratchpad/prof/counters.jl`.
Raw medians are in `scratchpad/prof/baseline.tsv`, paired results in
`scratchpad/prof/counter_comparison.txt`, and the full 2D CUPTI trace in
`scratchpad/prof/pull_2d_trace.json`. Runic formatted both scripts; a final
2D n=16 CUDA smoke run checked the driver, trace export and exact candidate
comparison after formatting.
Environment: Windows/WDDM, Julia 1.13.3, CUDA.jl 6.5.0,
KernelAbstractions 0.10.4, JustPIC 0.7.15 at `ac50925`, driver 617.42,
CUDA runtime 13.4. No package dependencies changed in this repository.

Median wall time in ms, 20 samples after two warmups per operation:

| Case | MQS RK2 | Linear RK2 | Pull mover (F7) | Colored mover |
|---|---:|---:|---:|---:|
| 2D n=256 Float32 | 2.156 | 0.353 | 1.827 | 6.737 |
| 2D n=256 Float64 | 3.282 | 3.026 | 2.524 | 15.903 |
| 3D n=64 Float32 | 4.904 | 3.198 | 3.687 | 22.911 |
| 3D n=64 Float64 | 32.701 | 29.458 | 8.554 | 35.736 |

Initial active counts: 1,560,600 (2D) and 8,001,504 (3D). These are the
actual quadrant-balanced counts, not requested `nxcell * ncells` estimates.
Every timed operation restores its input snapshot outside the timer. The
2D samples have display/clock noise (Float32 MQS ranged 1.391-6.310 ms);
these are preliminary device baselines, not precision speedup claims.
CUDA event intervals are recorded too, but include host gaps between
launches; they must not be treated as summed kernel-busy time.

On the 2D Float32 snapshot, CUPTI records 1 classification, 8 pull and
10 clear launches; 23 device-to-host copies and 9 device allocations
(including validation/reduction work). Kernel-busy times are 0.237 ms
classification, 0.737 ms pull and 0.486 ms clear in that instrumented run.
The 2D Float64 snapshot takes 11 rounds; both 3D snapshots take 2.
Do not compare the instrumented trace's wall time with unprofiled medians.
`CUDA.@metrics` was attempted: the current host also denies performance
counter access (`ERR_NVGPUCTRPERM`). CUPTI activity tracing works; no driver
settings were changed.

Validation uses exact particle IDs: every survivor retains its original
coordinates and carried ID, IDs are unique, and empty coordinate slots are
NaN. Locality is checked with a two-ULP-scale allowance for device FMA vs
host range materialization, with such cases counted explicitly (5 in the
3D Float32 case, also present with colored sweeps; none in Float64).
Pull and colored sweeps retain identical survivor IDs in 3D. In overflowing
2D cells they retain different IDs but the same count; matching counts alone
does not prove identical selection. Counter experiments must match **F7
pull** exactly, including slots, NaNs, fields and survivors.

#### F8: packed counter readbacks (evaluated against F7; implemented after F9, see below)

Prototype stores five flags/counters in one Int32 buffer, zeros pending and
progress together, and copies the buffer to the host once per round. It
uses the existing classification/pull/clear kernels and round schedule.
Alternating A/B order, same snapshots, 40 pairs:

| Case | F7 ms | Packed ms | Speedup |
|---|---:|---:|---:|
| 2D n=256 Float32 | 1.816 | 1.632 | 1.113x |
| 2D n=256 Float64 | 2.401 | 2.282 | 1.052x |
| 3D n=64 Float32 | 3.493 | 3.441 | 1.015x |
| 3D n=64 Float64 | 8.216 | 8.055 | 1.020x |

Exact index/coordinate/field equality with F7 passed for all four cases.
An 80-pair repeat reduced the 2D Float32 gain to 1.038x
(1.895 -> 1.826 ms); Float64 was 1.070x (2.406 -> 2.249 ms).
Defer promotion: 3D is neutral, and the small 2D gain is not yet stable
enough to justify adding array views to the cross-backend path. No source
change or production-test claim is made for this prototype.

Reproduce in an isolated environment containing the local JustPIC checkout,
CUDA, KernelAbstractions, CellArrays, CellArraysIndexing and NVTX:

```julia
include("scratchpad/prof/driver.jl")
c = GPUProfile.run_case(2, 256, Float32; samples=20, profile=true)
include("scratchpad/prof/counters.jl")
compare_counters(c; samples=80)
```

**Checkpoint that led to F9:** profile the 3D Float64 classification kernel (4.802 ms
of the split sample), especially coordinate lookup/range arithmetic, and
repeat on a refined vector grid to separate range arithmetic from memory
traffic. Also expand the 2D paired counter experiment across seeds and
n=1024 before reconsidering F8. Keep baseline snapshots fixed within each
A/B comparison; device RNG initialization can vary across fresh setups.
The current driver uses exact floating-point IDs and explicitly rejects
layouts exceeding `maxintfloat(T)` total slots. The n=1024 Float32 expansion
therefore needs a wider ID encoding or a separate validation pass before
timing; do not silently allow duplicate IDs.
The F9 production change below passed those CPU/CUDA gates.

### F9: cache classification bounds per cell (implemented)

`classify_moves_kernel!` previously repeated the seeded parent-cell lookup
for every live slot. On uniform Float64 ranges, repeated node computation
dominates classification. The kernel now computes its cell bounds once:

- Uniform ranges cache four nodes, covering the source and its immediate
  neighbors. Half-open comparisons resolve the usual stay/one-cell moves.
- Vector grids cache only the parent bounds. A non-staying particle uses
  the existing bisection. Caching four nodes on vector grids was measured
  and rejected (about 6% slower in 3D Float64 classification).
- Long jumps and edge cases use the original `half_open_cell` fallback.
  Periodic movement still uses colored sweeps. Pull/clear, full-cell drop
  handling and the round schedule are unchanged.

Classification prototype, alternating 40-pair comparisons on identical
advected snapshots (GPU event intervals in ms):

| Case | F7 | Cached | Speedup |
|---|---:|---:|---:|
| 3D n=64 Float64 ranges | 5.185 | 1.961 | 2.644x |
| 2D n=256 Float64 ranges | 0.708 | 0.243 | 2.913x |
| 3D n=64 Float32 ranges | 1.117 | 0.974 | 1.146x |
| 2D n=256 Float32 ranges | 0.199 | 0.153 | 1.300x |
| 3D n=64 Float64 vectors, parent bounds only | 1.313 | 1.005 | 1.306x |

All code arrays, outgoing masks, far and moving flags matched exactly.
Vector measurements here use materialized versions of the uniform grid;
they isolate node arithmetic, rather than constituting a refined-grid
application benchmark.

**Complete public mover**, 60 alternating pairs with snapshot restoration
outside the timer, F7 from `ac50925` versus the working-tree production change:

| Case | F7 ms | F9 ms | Speedup |
|---|---:|---:|---:|
| 3D n=64 Float64 | 8.413 | 5.469 | 1.538x |
| 2D n=256 Float64 | 2.452 | 1.957 | 1.253x |
| 3D n=64 Float32 | 3.550 | 3.454 | 1.028x |
| 2D n=256 Float32 | 1.783 | 1.724 | 1.034x |

Exact occupancy, slot placement, coordinate and carried-field equality
(including NaNs) passed for every case. Treat the small Float32 differences
as neutral. Local comparison code: `scratchpad/prof/classify.jl` and
`scratchpad/prof/baseline_move.jl`; the latter loads the F7 mover from git
`ac50925ee3b0dd4419446578687ffdc29a9a2bf8` into an isolated reference module.

Regression coverage added to `test/helpers_move_particles.jl` and registered
inside the existing 2D movement testset. It compares cached lookup with the
unchanged bisection on device, for every seed, vertices and adjacent floating
values, midpoints, long jumps and clamped endpoints, using uniform and
nonuniform grids. Production change: `src/Particles/move_pull.jl` only.

Validation passed: selected movement, periodic and nonperiodic ghost,
coloring, full-destination, cycle/drop, and converging-jump testsets from
`test_2D.jl`/`test_3D.jl`, on CPU and CUDA in both precisions. Local runner:
`include("scratchpad/prof/movement_gates.jl"); movement_gates()`.
All eight backend/precision/dimension configurations passed. The first tool
call hit its default execution timeout; the longer CPU run's result was
recovered from the Julia session log after the tool transport's five-minute
response limit. CUDA fixtures need the existing scalar-access allowance;
the local runner imports CUDA into Main so `helpers_backend.jl` enables it
as intended. The new regression kernel itself uses no device scalar access.
Saved gate logs: `scratchpad/prof/cpu_movement_gates.txt` and
`scratchpad/prof/cuda_movement_gates.txt`. Raw timing comparisons:
`scratchpad/prof/f9_results.tsv`. Runic formatting and `git diff --check`
pass. AMDGPU/Metal hardware is unavailable on this host; no results are
claimed for those devices. This is a focused movement gate, not a full
package-suite result.

**Next checkpoint after F9:** profile MQS advection again on this GPU,
separating velocity-grid node arithmetic/lookup from interpolation and
coordinate writes. In the initial 3D Float64 baseline MQS took 32.7 ms,
well above the now-improved mover. Expand the paired size/seed matrix
before pursuing another launch-layout or shared-memory change.

## Status after F9 and F8

A second, independent F9 gate run (isolated environment that develops
this checkout and adds CUDA, KernelAbstractions, Test, CellArrays,
CellArraysIndexing and StaticArrays, since the repository Manifest's
KernelAbstractions clashes with the global CUDA) also passed on CPU
(4 threads) and CUDA, both precisions, 2D and 3D.

### Mover split after F9

Median of 10 instrumented calls (`split_pull!` in the driver), same
seeded advected snapshots; wall time from separate unprofiled calls, ms:

| Case | Wall | Classify | Pull | Clear | Rounds (pull/clear) | Dropped |
|---|---:|---:|---:|---:|---:|---:|
| 2D n=256 Float32 | 1.773 | 0.168 | 0.769 | 0.347 | 10/12 | 1042 |
| 2D n=256 Float64 | 2.052 | 0.278 | 0.852 | 0.405 | 10/12 | 1090 |
| 3D n=64 Float32 | 4.270 | 1.039 | 1.506 | 0.797 | 2/2 | 0 |
| 3D n=64 Float64 | 5.327 | 2.133 | 2.151 | 1.199 | 2/2 | 0 |

The pull kernel is now the largest mover cost in every case. In 2D, chains
of full cells take 10 level-0 rounds plus two drop rounds, and host work
between launches is about a quarter of the wall time. In 3D,
classification is still several times above a bandwidth estimate for
reading the coordinates (no counter access to confirm the limiter).

### F8 promoted

The packed-counter prototype (`counters.jl`) was rerun against F9 with
three seeds (391, 7, 2026) and at n=1024, 60 alternating pairs each, exact
equality with F9 in every run. Every one of the 15 comparisons was faster:
2D n=256 1.054-1.097x (both precisions), 2D n=1024 1.009-1.033x,
3D n=64 Float32 1.046-1.061x.

The production version in `src/Particles/move_pull.jl` uses no array
views: one `Int32` buffer of five entries indexed by the constants `FAR`,
`MOVING`, `PENDING`, `PROGRESS` and `DROPPED`, with a host copy that is
written back once per round (one copy to the device, one back) instead of
two `fill!` launches and two `maximum` reductions. `raise!(x, i, value)`
in `move_safe.jl` gained an index; `raise!(x, value)` keeps its meaning.
Against an isolated F9 reference module, 60 alternating pairs, exact
occupancy/coordinate/field equality and identical drop counts:

| Case | F9 ms | F8 ms | Speedup |
|---|---:|---:|---:|
| 2D n=256 Float32 | 1.656 | 1.541 | 1.075x |
| 2D n=256 Float64 | 2.022 | 1.919 | 1.054x |
| 2D n=1024 Float32 | 13.107 | 13.002 | 1.008x |
| 2D n=1024 Float64 | 16.373 | 16.097 | 1.017x |
| 3D n=64 Float32 | 3.369 | 3.294 | 1.023x |
| 3D n=64 Float64 | 5.423 | 5.300 | 1.023x |
| 3D n=128 Float32 | 22.217 | 21.962 | 1.012x |

The gain is small but has the same sign in all 22 paired comparisons, and
the change removes four device allocations per call. `counters.jl` is now
superseded; `driver.jl`'s `split_pull!` follows the new kernel signatures.
Movement gates with F8, CPU (4 threads) and CUDA, Float32 and Float64,
2D and 3D: 18776 of 18776 tests pass across the eight configurations.
Runic (LF copies) and `git diff --check` pass.

### Rejected after F9

Method redefinition inside a session, ABAB order, 30 samples per block,
bit-identical results:

- Read the shared flag before the atomic `max` in `raise!` (skip the
  atomic when already set): 0.995-1.059x. Within noise.
- Resume `next_free_slot` from the previous slot instead of iterating
  from slot 1: 0.981-1.014x. Within noise.

### Mover follow-ups (after the MQS checkpoint above)

1. Pull in 3D: two full-grid launches cost 1.5-2.2 ms. Measure how many
   neighbours send to a typical cell and how many code slots each scan
   reads, then decide between a per-cell incoming count written during
   classification and a different scan order.
2. Rounds in 2D: after the first round only a few cells are active, but
   each round still launches pull and clear over the whole grid.
   Evaluate an active-cell list (or early exit by a per-row flag) against
   the extra atomics it needs.
3. The colored path regressed 3-5% in 3D with F5 (branchless free-slot
   scan with `max_xcell` = 50). It now runs only for periodic or long-jump
   moves; consider keeping the early exit when `max_xcell` is large.

## Original investigation and RTX 3080 evidence

Target: the hot loop of `scripts/temperature_advection3D.jl` (3D) and
`scripts/temperature_advection.jl` (2D):

```julia
advection_MQS!(particles, RungeKutta2(), V, dt)
move_particles!(particles, particle_args)
```

Historical hardware: NVIDIA GeForce RTX 3080 (10 GB, 68 SMs, ~100k resident threads).
Tools: CUDA.jl, `nsys`, `ncu`.

Note: `scripts/temperature_advection3D.jl` now calls
`advection_MQS!(particles, RungeKutta2(), V, dt)`, and
`scripts/temperature_advection.jl` calls
`advection!(particles, RungeKutta4(), V, dt)`. The profiling driver calls
`advection_MQS!(particles, RungeKutta2(), V, dt)` explicitly in both
dimensions, with `advection!` as a baseline.

## Phase 0: Profiling driver

Write one driver script (outside `src/`, not committed) that:

- reuses the setup of the 2D and 3D scripts, with GLMakie and plotting removed;
- takes parameters `dim ∈ {2, 3}`, `n`, element type `T ∈ {Float64, Float32}`,
  `nxcell` / `max_xcell`, and the advection function (`advection_MQS!` or
  `advection!`);
- runs one warmup step so compilation is excluded;
- wraps each call in an NVTX range (`CUDA.NVTX.@range "advect"` /
  `CUDA.NVTX.@range "move"`).

The `advection!` baseline isolates the cost of the MQS reconstruction.

## Phase 1: Timing breakdown

1. Time each call separately with `CUDA.@sync` and BenchmarkTools.
2. Run `CUDA.@profile` over ~10 steps. Per call, record:
   - kernel count, GPU time, host API time;
   - the gap between kernel-busy time and wall time (launch plus sync
     overhead);
   - number of device-to-host copies (the `maximum(counter)` reads in
     `sweep_cells!`);
   - device allocations (the `KernelAbstractions.zeros` counters allocated in
     every sweep).
3. Run under Nsight Systems for a timeline:

   ```sh
   nsys profile --trace=cuda,nvtx -o timeline julia --project driver.jl
   ```

   Check for GPU idle gaps between the color launches of `move_particles!`
   (27 per sweep in 3D, 9 in 2D).

## Phase 2: Per-kernel analysis with `ncu`

```sh
ncu --kernel-name regex:advection_kernel_MQS --launch-skip 2 --launch-count 1 --set full -o adv julia --project driver.jl
ncu --kernel-name regex:move_particles_ps   --launch-skip 30 --launch-count 3 --set full -o mov julia --project driver.jl
```

Run both for 2D and 3D. Metrics to read:

- registers per thread, theoretical vs achieved occupancy;
- FP64 pipe utilization;
- DRAM and L2 throughput as % of peak;
- sectors per request (coalescing);
- warp execution efficiency (divergence);
- top warp stall reasons.

## Hypotheses

### `advection_MQS!`

| # | Suspect | Confirming signal | Fix |
|---|---|---|---|
| A1 | FP64 on a GeForce GPU: the RTX 3080 runs FP64 at 1/64 of FP32 speed, and the scripts are all Float64. | FP64 pipe saturated; Float32 run is about 5–20× faster. | Run in Float32 (code is eltype-generic per `.agents/backends.md`). Hardware limit, not a code defect. |
| A2 | One thread per cell loops serially over `max_xcell` slots; `doskip` causes divergence (~50% of slots empty). | Low warp efficiency; time scales with `max_xcell`, not particle count. | One thread per (slot, cell) via ndrange `(nslots, ncells...)`, or one workgroup per cell with one thread per slot. The SoA CellArray layout keeps coalescing. |
| A3 | Register pressure: RK2 × N components × MQS stencil, with NTuple inlining. | > 128 registers/thread; occupancy < 25%. | Interpolate one component at a time, `@noinline` per-component MQS, or cap registers. |
| A4 | Redundant stencil reads: every particle in a cell reloads the same velocity nodes. | High L1/L2 traffic, low DRAM traffic. | Stage the cell's velocity stencil in `@localmem` once per workgroup (needs the A2 layout). |
| A5 | Bounds checks and the `Inf` / `check_local_limits` branches. | Branch-heavy SASS; small gain with `--check-bounds=no`. | Low priority; act only if the profile shows it. |

### `move_particles!`

| # | Suspect | Confirming signal | Fix |
|---|---|---|---|
| M1 | Launch and sync overhead: each sweep makes 3ᴺ color launches (27 in 3D, 9 in 2D), each `launch!` synchronizes, each sweep allocates 5 device counters and does 5 blocking device-to-host `maximum` reads. | `nsys` idle gaps; wall time ≫ sum of kernel times. | Skip per-launch sync inside the color loop (same stream keeps ordering). Pack the counters in one `Int32` array of length 5, allocated once; one device-to-host copy per sweep. |
| M2 | Under-filled GPU: each color launch has only (n/3)ᴺ threads (≈ 9k at n = 64 in 3D, ≈ 7k at n = 256 in 2D), each looping over all slots serially. | Very low achieved occupancy; latency-bound stalls. | Keep coloring, but one workgroup per cell with one thread per slot; or the pull scheme (M4). |
| M3 | Strided, uncoalesced access (same-color cells are 3 apart) and linear scans in `find_free_memory` / `settled`. | High sectors per request. | Per-cell free-slot count or bitmask; scan `settled` only when the destination is full. |
| M4 | Structural: coloring forces 3ᴺ serialized phases. | Remains after M1–M3. | Pull scheme: (1) per-particle kernel stores the destination offset (`Int8`) in one fully parallel launch; (2) per-cell kernel gathers incoming particles from its 3ᴺ − 1 neighbors into its own free slots (race-free, each thread writes only its own cell); (3) kernel clears moved-out slots. Risk: the full-destination and drop semantics (#378) and the occupancy invariant must be preserved exactly; needs a design pass and the existing `move_particles!` tests on CPU and CUDA. |
| M5 | Host-side checks: `check_active_coordinates`, `check_particle_fields`, `physical_domain_limits` (`extrema`), each possibly a reduction plus sync. | Visible in the `CUDA.@profile` host API list. | Cache or hoist them out of the hot path while keeping fail-fast behavior. |

## 2D-specific points

- The 2D case has fewer cells per launch for equal `n`: 9 color launches per
  sweep, each with (n/3)² threads. At n = 256 that is ≈ 7k threads, so the
  GPU is badly under-filled. Expect M1 and M2 to dominate in 2D up to
  n ≈ 1024 (≈ 116k threads per launch).
- Launch overhead (M1) is a larger fraction of total time in 2D because each
  kernel does less work.
- `advection_MQS!` in 2D has a smaller stencil and fewer components, so A3
  (register pressure) matters less and A2 (divergence) relatively more.
- The 2D script uses `nxcell, max_xcell, min_xcell = 24, 30, 12`; the 3D script
  uses `25, 50, 10`. Compare both at matching `max_xcell` to separate
  dimension effects from slot-count effects.
- Measure the crossover `n` where the 2D launch becomes throughput-bound
  rather than latency-bound; this gives the problem size below which GPU
  execution is not worthwhile.

## Phase 3: Experiment matrix

| Dimension | `n` | `T` | `max_xcell` |
|---|---|---|---|
| 2D | 256, 512, 1024, 2048 | Float64, Float32 | 30, 50 |
| 3D | 64, 128 | Float64, Float32 | 30, 50 |

For each configuration, record ms/step for each call and the `ncu` headline
metrics. Separate hardware limits (A1) from algorithmic limits (A2, M1, M2).

## Phase 4: Fix order

1. **M1**: sync and counter cleanup. Small, safe, likely a large gain at small
   `n` and in 2D.
2. **A1**: Float32 comparison. No code change; benchmark data and
   documentation only.
3. **A2 / M2**: thread-per-slot layout. Medium-size, kernel-local change.
4. **A4, A3**: only if `ncu` still shows memory or register limits after step 3.
5. **M4**: pull-scheme redesign on a separate branch, after a design pass.

Validate every fix with the `test/` suite on CPU and with
`--backend=CUDA`, in both 2D and 3D, and check the particle occupancy
invariant (`src/Particles/move_safe.jl`).

## Expected outcome

- `move_particles!`: M1 and M2 dominate, more so in 2D. The GPU mostly waits
  on launches and syncs.
- `advection_MQS!`: FP64 throughput limits first on this GPU, then divergence
  from the serial slot loop.

## Findings (RTX 3080, CUDA.jl 6.5, KernelAbstractions 0.10.4, Julia 1.13)

Driver and experiment scripts: `scratchpad/prof/driver.jl` (session-local).
`ncu` is unavailable until GPU performance counters are enabled for non-admin
users (`ERR_NVGPUCTRPERM`); results below come from `CUDA.@profile`,
`CUDA.@elapsed`, PTX inspection, and kernel ablation.

### Baseline (ms per call, median of 20 steps)

| Case | `advection_MQS!` | `advection!` | `move_particles!` |
|---|---|---|---|
| 3D n=64 Float64 | 191 | 1359 | 73 |
| 3D n=64 Float32 | 19 | 130 | 42 |
| 3D n=128 Float64 | 1659 | 12423 | 376 |
| 3D n=128 Float32 | 166 | 1254 | 220 |
| 2D n=256 Float64 | 18 | 153 | 34 |
| 2D n=256 Float32 | 1.3 | 11 | 13 |
| 2D n=1024 Float64 | 325 | 2646 | 269 |
| 2D n=1024 Float32 | 21 | 183 | 91 |
| 2D n=2048 Float64 | 1413 | 11488 | 936 |
| 2D n=2048 Float32 | 90 | 836 | 304 |

### F1: bounds-checked store loop in advection kernels (fixed)

`advection!` was 7–9× slower than `advection_MQS!` although it does less
work. Cause: the coordinate write-back

```julia
for k in 1:N
    CAI.@index p[k][ipart, I_inner...] = pᵢ_new[k]
end
```

indexes the coordinate tuple with a runtime `k` and keeps a bounds check; the
loop is then not unrolled. `advection_MQS!` and the MarkerChain kernel hid this
with `@inbounds`. Replacing the loop with the existing unrolled
`fill_particle!(p, pᵢ_new, ipart, I_inner)` keeps the bounds checks and gives
the same speed as the `@inbounds` version:

| Case | `advection!` before | after |
|---|---|---|
| 3D n=64 Float32 | 121 ms | 15 ms |
| 3D n=64 Float64 | 1359 ms | 165 ms |
| 2D n=256 Float32 | 11.7 ms | 1.05 ms |
| 2D n=1024 Float64 | 2646 ms | 254 ms |

Applied to `advection.jl`, `advection_LinP.jl`, `advection_MQS.jl`, and
`MarkerChain/Advection/advection.jl`.

### F2: FP64 throughput (A1 confirmed)

Float64 is 9–16× slower than Float32 for advection on this GeForce GPU, and
1.7–3× slower for `move_particles!`. PTX of the Float32 kernels contains no
`f64` instructions, so the Float32 path is clean.

### F3: `move_particles!` is latency-bound inside the color kernels

- Launch/sync overhead (M1) is small: in 3D n=64 the GPU is busy 42.3 ms of a
  43.4 ms call (36 launches, 13 device-to-host copies).
- Sweeps per call: 3D n=64 does 2 (27 + 9 color launches); 2D n=256 does 4
  (36 launches), because many cells reach `max_xcell` in this flow without
  cleanup (~1400 full cells; ~1000 particles dropped per call).
- Locating movers is cheap: one `containing_cell` pass over all particles
  takes 1.6 ms (3D n=64) as one launch, 3.7 ms as 27 color launches.
- Ablation of the first 3D sweep (32 ms): scan + jump logic 6 ms;
  `find_free_memory` +12 ms; slot writes +7 ms. In 2D, `settled` adds ~3 ms
  because full destinations are common.
- `find_free_memory` costs 0.26 ms for 8 lookups per cell when every cell runs
  concurrently, so its cost in the sweep comes from exposed memory latency:
  each color launch has ~10k threads (~10% of the GPU's resident capacity),
  and each thread walks its slots and the destination's slots serially.

Conclusion: M2 (under-filled GPU, serial per-thread chains) dominates;
M3 is where the latency accumulates. Fixes that add parallelism per launch
(workgroup per cell with one thread per slot, or the pull scheme M4) target
the real cost; M1 alone would save at most a few percent.

### F4: `advection_MQS!` cost is dominated by the parent-cell lookup (fixed in part)

Ablation of the MQS kernel, replacing only the interpolation function
(3D n=64, before the fix): constant velocity 0.6 ms; parent-cell lookup only
12.1 ms; lookup + corner gather 12.1 ms; full MQS 20.4 ms. Float64: lookup
147 ms of 174 ms. The interpolation arithmetic is cheap.

Cause: `find_parent_cell_bisection` is seeded with the particle's cell, but on
the staggered velocity grids the containing cell is `seed - 1` for 66% (3D) /
75% (2D) of lookups. A miss restarts a bisection over the whole axis
(~log₂ n iterations, `Int64` `div`, repeated `LinRange` `getindex` that
divides in Float64). The search now tests `seed`, `seed - 1`, `seed + 1`
before bisecting; results are unchanged because the bracket is unique.

Thread-per-slot launch layouts for the MQS kernel gave ≤ 25% in 3D Float32
and were slower in 2D; not pursued.

Registers (2D Float32, `ptxas -v`, sm_86): MQS kernel 99 registers, 176-byte
stack; `move_particles_ps!` 190 registers, 696-byte stack.

### F5: branchless `find_free_memory`

The early-exit scan serializes one dependent load per occupied slot. A
branchless scan returns the same slot (verified bit-identical) and speeds
`move_particles!` up 12–15% in 2D; neutral in 3D.

### A/B against `main` (ms per call, median of 10 steps, same session order)

| Case | MQS main | MQS now | linear main | linear now | move main | move now |
|---|---|---|---|---|---|---|
| 3D n=64 Float32 | 19.6 | 15.7 (1.25×) | 123 | 10.4 (11.8×) | 71.8 | 72.9 (1.0×) |
| 3D n=64 Float64 | 199 | 107 (1.85×) | 1291 | 82.5 (15.6×) | 133 | 124 (1.07×) |
| 3D n=128 Float32 | 168 | 123 (1.36×) | 1203 | 83.7 (14.4×) | 372 | 384 (0.97×) |
| 2D n=256 Float32 | 1.17 | 1.36 (0.86×) | 10.4 | 0.95 (10.9×) | 23.3 | 19.9 (1.17×) |
| 2D n=1024 Float32 | 20.3 | 16.4 (1.24×) | 169 | 11.8 (14.4×) | 170 | 132 (1.29×) |
| 2D n=1024 Float64 | 316 | 156 (2.02×) | 2334 | 105 (22.3×) | 550 | 371 (1.48×) |

`init_particles` draws from the device RNG, so each run starts from a
different particle set; differences below ~5% (and the 2D n=256 MQS entry,
a ~1 ms kernel) are within run-to-run noise.

### Remaining bottleneck before F6-F7 (historical)

`move_particles!` is now the largest cost (3D n=64 Float32: 73 ms vs 16 ms
for `advection_MQS!`). The cost is latency inside the colored launches, not
launch overhead; the remaining levers are structural (M2: more threads per
cell; M4: pull scheme) and change which free slot a particle lands in.

### F6: pull scheme for `move_particles!` (M4, implemented)

`src/Particles/move_pull.jl`. Used when no direction is periodic and every
particle is at most one cell from its parent cell; otherwise the colored
sweeps run unchanged. One classification launch writes an `Int8` code per
slot (stay / out of domain / too far / offset to destination), then rounds of
two full-grid launches: each cell pulls the particles bound for it into the
slots that were free when the round started, then each cell removes the
particles that were pulled away or left the domain. A stalled round first
drops only particles bound for full cells with no waiting particles of their
own, then resumes; a second stall drops the rest (cycles of full cells).

Checked against the colored sweeps from identical snapshots: same number of
surviving particles in all cases (3D n=64 Float32/Float64, 2D n=256/1024),
no particle outside its cell, empty slots NaN. Slot placement differs, which
the tests allow (they check slot sets, not exact slots). Typical calls take 3
rounds (3D n=128: classify 38 ms, pull 78 ms, clear 8 ms).

### A/B against `main` after F1–F6 (ms per call, median of 10 steps)

| Case | MQS main → now | linear main → now | move main → now |
|---|---|---|---|
| 3D n=64 Float32 | 19.6 → 17.7 (1.1×) | 124 → 12.6 (9.8×) | 71.8 → 19.7 (3.6×) |
| 3D n=64 Float64 | 199 → 108 (1.8×) | 1289 → 83.9 (15.4×) | 133 → 29.2 (4.6×) |
| 3D n=128 Float32 | 168 → 139 (1.2×) | 1204 → 95.9 (12.6×) | 375 → 141 (2.7×) |
| 2D n=256 Float32 | 1.17 → 1.40 (0.8×) | 10.3 → 0.95 (10.8×) | 23.0 → 3.46 (6.6×) |
| 2D n=1024 Float32 | 20.3 → 17.2 (1.2×) | 167 → 12.0 (13.9×) | 167 → 42.2 (4.0×) |
| 2D n=1024 Float64 | 316 → 152 (2.1×) | 2334 → 102 (22.8×) | 549 → 62.4 (8.8×) |

This run overlapped the CPU test suite, so host-side timings carry extra
noise; the MQS column is ~10% above the previous quiet run.

### F7: pruning the pull rounds

`outgoing[I]` (`UInt32`) holds one bit per destination offset of the waiting
particles of cell `I`, plus a bit for particles that left the domain. The
pull kernel skips neighbors with no bit for it; the clear kernel skips cells
whose mask is zero; free slots are searched only when a particle arrives.

`move_particles!`, same snapshots, colored sweeps → pull scheme:

| Case | sweeps | pull | speedup |
|---|---|---|---|
| 3D n=64 Float32 | 40.4 ms | 9.4 ms | 4.3× |
| 3D n=64 Float64 | 62.1 ms | 17.4 ms | 3.6× |
| 2D n=256 Float32 | 9.5 ms | 2.25 ms | 4.2× |
| 2D n=1024 Float32 | 69.8 ms | 22.0 ms | 3.2× |

Kernel split, 3D n=64 Float32: classify 4.2 ms, pull 4.0 ms (2 rounds),
clear 0.7 ms. 2D n=1024 Float32: 11 pull rounds (chains of full cells
resolve one link per round), classify 7.3 ms, pull 9.9 ms, clear 4.1 ms.
A thread-per-slot classify kernel was not faster.

New test `check_full_destination_chain` (2D and 3D): particles bound for a
settled full cell are dropped and free their source for an upstream particle.

## Historical validation status (RTX 3080 session)

- CPU: `test_3D`, `test_integrators`, `test_markerchain_2D`, `test_refined_grid`
  pass. `test_2D` passes every testset through `Forced injection 2D` (all move
  tests included) but crashes intermittently (exit 139, sometimes
  `GC error (probable corruption)` during type inference) in `Pure shear 2D`.
  The crash occurs on `main` too: running only that testset, 1/5 runs crashed
  on `main` and 2/5 on this tree; `--check-bounds=yes` raises no `BoundsError`.
  The testset defines a `@kernel` inside the `@testset` body.
- CUDA: `test_2D`, `test_3D`, `test_markerchain_2D`, `test_refined_grid`,
  `test_integrators` pass with `CUDA.allowscalar(true)`. Without it, the same
  four files stop at a scalar-indexing error in test setup code (or
  `MarkerChain/resample.jl:17`), identically on `main`.

# GPU profiling plan: `advection_MQS!` and `move_particles!`

Target: the hot loop of `scripts/temperature_advection3D.jl` (3D) and
`scripts/temperature_advection.jl` (2D):

```julia
advection_MQS!(particles, RungeKutta2(), V, dt)
move_particles!(particles, particle_args)
```

Hardware: NVIDIA GeForce RTX 3080 (10 GB, 68 SMs, ~100k resident threads).
Tools: CUDA.jl, `nsys`, `ncu`.

Note: `scripts/temperature_advection3D.jl` on disk calls `advection!`
(linear), and `scripts/temperature_advection.jl` calls
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

### Remaining bottleneck

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

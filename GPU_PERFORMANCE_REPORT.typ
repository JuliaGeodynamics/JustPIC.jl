// JustPIC GPU speedups: adm/gpu-perf against main on an RTX 5070 Ti.
// Compile with: typst compile GPU_PERFORMANCE_REPORT.typ

#let accent = rgb("#0f6e7a")
#let accent-soft = rgb("#d8eef0")
#let muted = rgb("#5b6670")
#let line-color = rgb("#d9dee3")
#let bar-main = rgb("#b9c2ca")
#let good = rgb("#1d7a45")

#set document(title: "JustPIC GPU Speedups", date: datetime(year: 2026, month: 10, day: 29))
#set page(paper: "a4", margin: (x: 2cm, y: 2.2cm), numbering: "1")
#set text(font: ("Segoe UI", "Libertinus Serif"), size: 10pt, lang: "en")
#set par(justify: false, leading: 0.65em)
#show raw: set text(font: ("Consolas", "DejaVu Sans Mono"), size: 0.92em)
#show raw.where(block: false): box.with(fill: accent-soft, inset: (x: 2pt), outset: (y: 2pt), radius: 2pt)
#show heading: set text(font: ("Georgia", "Libertinus Serif"))
#show heading.where(level: 1): it => {
  v(1em)
  line(length: 100%, stroke: 0.5pt + line-color)
  v(0.3em)
  text(size: 15pt, it)
  v(0.3em)
}
#show heading.where(level: 2): set text(size: 11.5pt)

#set table(stroke: (x, y) => if y > 0 { (top: 0.4pt + line-color) }, inset: (x: 6pt, y: 5pt))
#show table.cell.where(y: 0): set text(fill: muted, weight: "semibold", size: 8pt)
#show table: set text(size: 8.8pt, number-type: "lining", number-width: "tabular")

#let x(v) = text(fill: good, weight: "semibold")[#v×]

#let change(title, tag, impact, before, change, why, measured) = block(
  width: 100%,
  stroke: 0.5pt + line-color,
  radius: 4pt,
  inset: 12pt,
  breakable: true,
)[
  #text(font: ("Georgia"), size: 11.5pt, weight: "semibold", title)
  #h(8pt) #text(size: 8pt, fill: muted, font: ("Consolas"), tag)
  #h(8pt) #box(fill: accent-soft, inset: (x: 5pt, y: 2pt), radius: 2pt, text(size: 8.5pt, fill: good, weight: "semibold", impact))
  #v(4pt)
  #grid(
    columns: (70pt, 1fr),
    row-gutter: 8pt,
    column-gutter: 10pt,
    text(size: 8pt, fill: muted, weight: "semibold")[BEFORE], before,
    text(size: 8pt, fill: muted, weight: "semibold")[CHANGE], change,
    text(size: 8pt, fill: muted, weight: "semibold")[WHY FASTER], why,
    text(size: 8pt, fill: muted, weight: "semibold")[MEASURED], measured,
  )
]

#text(size: 8pt, fill: accent, tracking: 0.08em, font: ("Consolas"))[ADM/GPU-PERF VS MAIN · RTX 5070 TI]
#v(-0.4em)
#text(font: ("Georgia"), size: 24pt, weight: "semibold")[JustPIC GPU Speedups]

#text(size: 11pt)[
  The hot loop of the advection scripts, `advection_MQS!` followed by `move_particles!`, runs 2.8× to 5.6× faster per time step than on `main`. Most of the gain comes from two changes: a new pull scheme for moving particles between cells, and a cheaper parent-cell lookup during advection.
]

#text(size: 9pt, fill: muted)[
  Measured 2026-10-29. Baseline `main` at `6608404`; branch `adm/gpu-perf` at `ac50925` plus the uncommitted cached-bounds change in `src/Particles/move_pull.jl`.
]

#v(0.5em)
#grid(
  columns: (1fr,) * 4,
  column-gutter: 8pt,
  ..(
    ("2.8–5.6×", [faster time step (MQS advection + move)]),
    ("3.9–10.6×", [faster `move_particles!`]),
    ("1.6–3.1×", [faster `advection_MQS!`]),
    ("2.0–3.2×", [faster `advection!` (linear)]),
  ).map(((v, l)) => block(width: 100%, stroke: 0.5pt + line-color, radius: 4pt, inset: 9pt)[
    #text(size: 15pt, weight: "semibold", fill: accent, font: ("Consolas"), v) \
    #text(size: 8.5pt, fill: muted, l)
  ]),
)

= Gains against main

Each value is the mean of two medians, from two runs of 30 samples per operation in separate processes. Runs alternated between `main` and the branch. The two medians agree within 8% for every entry. All times are in milliseconds per call.

#table(
  columns: (auto,) + (1fr,) * 9,
  align: (left,) + (right,) * 9,
  table.header(
    [], table.cell(colspan: 3, align: center)[ADVECTION_MQS! (RK2)], table.cell(colspan: 3, align: center)[MOVE_PARTICLES!], table.cell(colspan: 3, align: center)[TIME STEP (SUM)],
    [CASE], [MAIN], [BRANCH], [GAIN], [MAIN], [BRANCH], [GAIN], [MAIN], [BRANCH], [GAIN],
  ),
  [2D n=256 Float32], [0.85], [0.35], x(2.5), [7.22], [1.70], x(4.2), [8.07], [2.05], x(3.9),
  [2D n=256 Float64], [8.77], [3.22], x(2.7), [19.36], [1.83], x(10.6), [28.13], [5.05], x(5.6),
  [2D n=1024 Float32], [9.39], [4.69], x("2.0"), [54.41], [13.96], x(3.9), [63.81], [18.65], x(3.4),
  [2D n=1024 Float64], [141.8], [46.6], x("3.0"), [125.8], [16.3], x(7.7), [267.6], [62.9], x(4.3),
  [3D n=64 Float32], [7.67], [4.85], x(1.6), [20.97], [3.29], x(6.4), [28.65], [8.14], x(3.5),
  [3D n=64 Float64], [70.8], [32.4], x(2.2), [36.06], [5.39], x(6.7), [106.8], [37.8], x(2.8),
  [3D n=128 Float32], [61.9], [36.9], x(1.7), [116.7], [23.7], x(4.9), [178.6], [60.6], x(2.9),
)

== Linear advection (`advection!`, RK2)

#table(
  columns: (auto,) + (1fr,) * 7,
  align: (left,) + (right,) * 7,
  table.header([CASE], [2D 256 F32], [2D 256 F64], [2D 1024 F32], [2D 1024 F64], [3D 64 F32], [3D 64 F64], [3D 128 F32]),
  [main], [0.57], [7.92], [9.34], [138.6], [6.15], [67.8], [53.4],
  [branch], [0.26], [2.97], [3.43], [43.2], [3.14], [29.3], [24.7],
  [gain], x(2.2), x(2.7), x(2.7), x(3.2), x("2.0"), x(2.3), x(2.2),
)

== Time step, main against branch

#let step-rows = (
  ("2D 256 F32", 8.07, 2.05),
  ("2D 256 F64", 28.13, 5.05),
  ("2D 1024 F32", 63.81, 18.65),
  ("2D 1024 F64", 267.63, 62.87),
  ("3D 64 F32", 28.65, 8.14),
  ("3D 64 F64", 106.84, 37.83),
  ("3D 128 F32", 178.59, 60.59),
)
#block(stroke: 0.5pt + line-color, radius: 4pt, inset: 10pt, width: 100%, breakable: false)[
  #text(size: 8pt, fill: muted)[
    #box(width: 8pt, height: 8pt, fill: bar-main, radius: 1pt) main (100%) #h(12pt)
    #box(width: 8pt, height: 8pt, fill: accent, radius: 1pt) branch, share of main time
  ]
  #v(4pt)
  #grid(
    columns: (75pt, 1fr),
    row-gutter: 7pt,
    column-gutter: 8pt,
    ..step-rows
      .map(((label, m, b)) => (
        text(size: 8pt, fill: muted, font: ("Consolas"), label),
        stack(
          spacing: 2pt,
          rect(width: 80%, height: 6pt, fill: bar-main, radius: 1pt),
          stack(
            dir: ltr,
            spacing: 4pt,
            rect(width: 80% * b / m, height: 6pt, fill: accent, radius: 1pt),
            text(size: 7pt, str(calc.round(100 * b / m)) + "%"),
          ),
        ),
      ))
      .flatten(),
  )
]
#text(size: 8.5pt, fill: muted)[Bars use one linear scale: the grey bar is the main time for that case, the teal bar the branch time.]

= What made the gains

Five changes are kept on the branch. They are listed by size of contribution. The IDs (F1 to F9) match `GPU_PROFILING_PLAN.md`.

#change(
  [1. Pull scheme for `move_particles!`],
  [F6 + F7 · src/Particles/move_pull.jl],
  [move 2.6–6.3×],
  [Particles moved with colored sweeps: 3#super[N] serialized launches per sweep (9 in 2D, 27 in 3D), so that neighbouring cells never write at the same time. Each launch covers only every third cell per direction, about 7k threads in 2D at n=256 and 9k in 3D at n=64. Each thread walks its slots and the destination's slots in sequence. Profiling showed the GPU busy during these launches but stalled on memory latency, because so few threads were resident to hide it.],
  [When no direction is periodic and no particle jumped more than one cell, three kinds of full-grid launches replace the sweeps. A classification kernel writes an `Int8` destination code per slot. A pull kernel lets each cell gather the particles bound for it into its own free slots. A clear kernel empties the slots that were pulled away. Each thread writes only its own cell, so no coloring is needed. F7 adds a `UInt32` mask per cell with one bit per destination, so pull skips neighbours that send nothing and clear skips untouched cells. Periodic or long-jump cases still use the sweeps.],
  [Every launch now covers all cells, so the GPU has about 3#super[N] times more threads in flight to hide latency, and two or three rounds replace up to 36 serialized launches.],
  [Same session and seeded snapshots, the branch's colored sweeps (F5 included) against the committed F7 pull mover: 2D n=1024 Float64 102.1 → 22.6 ms (4.5×); 3D n=64 Float32 22.3 → 4.2 ms (5.3×); 3D n=128 Float32 121.7 → 24.5 ms (5.0×). The range over all seven cases is 2.6× (2D n=256 Float32) to 6.3× (2D n=256 Float64).],
)

#change(
  [2. Neighbour check before bisection in the parent-cell lookup],
  [F4 · src/Utils.jl],
  [advection 1.6–3.2×],
  [`find_parent_cell_bisection` starts from the particle's own cell. On staggered velocity grids the containing cell is `seed - 1` for 66% (3D) to 75% (2D) of lookups. Every miss restarted a bisection over the whole axis: about log#sub[2] n iterations, each with an `Int64` division and a `LinRange` `getindex` computed in the element type.],
  [Test `seed`, `seed - 1` and `seed + 1` first, then fall back to the bisection. The bracket is unique, so results do not change.],
  [Ablation showed the lookup took most of the advection kernel; the interpolation arithmetic is cheap. With the neighbour check almost every lookup takes one or two comparisons. The gain is largest in Float64, where each node computation costs more on this GeForce GPU, whose FP64 rate is a small fraction of its FP32 rate.],
  [Main with only F1 applied against the branch, same session: 2D n=1024 Float64 MQS 139.9 → 46.6 ms; 3D n=64 Float64 MQS 68.0 → 32.4 ms; 3D n=128 Float32 linear 52.0 → 24.7 ms. Every advection gain in the table above, except part of the smallest case, comes from this change.],
)

#change(
  [3. Cached cell bounds in move classification],
  [F9 · src/Particles/move_pull.jl (uncommitted)],
  [move 1.25–1.54× in Float64],
  [`classify_moves_kernel!` called `containing_cell` for every live slot, which recomputed the cell's grid nodes for each particle. In Float64 this node arithmetic dominated classification.],
  [Each thread loads its cell bounds once. Uniform ranges cache four nodes (the cell and both neighbours), so the common stay and one-cell moves are resolved with half-open comparisons. Vector grids cache only the parent bounds and bisect for moving particles. Long jumps and edge cases use the original `half_open_cell` fallback.],
  [Node computation moves out of the per-particle loop: one computation per cell instead of one per particle, about 25–50 per cell.],
  [Full mover, 60 alternating pairs against F7: 3D n=64 Float64 8.41 → 5.47 ms (1.54×); 2D n=256 Float64 2.45 → 1.96 ms (1.25×). Float32 is neutral (1.03×). Across sessions, 2D n=1024 Float64 drops from 22.6 to 16.3 ms (1.39×). Output is bit-identical to F7: occupancy, slots, coordinates and fields.],
)

#change(
  [4. Branchless free-slot search],
  [F5 · src/Particles/move_safe.jl],
  [colored path: 2D 1.13–1.22×, 3D 0.95×],
  [`find_free_memory` returned at the first free slot, so each slot read depended on the previous one.],
  [Scan every slot with `ifelse` and no early exit. It returns the same slot.],
  [The slot loads are independent, so they issue together instead of one at a time.],
  [Applies to the colored sweeps, which now run only for periodic or long-jump moves. Same session, main against the branch's colored sweeps: 1.13–1.22× faster in 2D (for example 2D n=1024 Float64 124.1 → 102.1 ms), but 3–5% slower in 3D (3D n=128 Float32 115.8 → 121.7 ms). In 3D the scan reads all 50 slots even when an early slot is free. The RTX 3080 showed the same pattern: 12–15% in 2D, neutral in 3D. A follow-up could keep the early exit for large `max_xcell`.],
)

#change(
  [5. Unrolled coordinate write-back],
  [F1 · four advection kernels],
  [≤1.5× here, 7–9× on RTX 3080],
  [The advection kernels wrote the new coordinates in a `for k in 1:N` loop that indexed the coordinate tuple with a runtime `k`. The loop kept a bounds check and was not unrolled; `advection_MQS!` hid it with `@inbounds`.],
  [Use the existing unrolled `fill_particle!` helper in `advection.jl`, `advection_LinP.jl`, `advection_MQS.jl` and the MarkerChain kernel. Bounds checks stay on.],
  [With a compile-time index the tuple access resolves statically and the store loop unrolls.],
  [On the RTX 3080 toolchain `advection!` went from 7–9× slower than `advection_MQS!` to the same speed. On this GPU, with CUDA.jl 6.5, F1 alone changes every case by 8% or less except 2D n=256 Float32 MQS (0.85 → 0.67 ms, a sub-millisecond kernel with noisy timings). It is kept because it removes an `@inbounds` and guards against the regression on other compilers.],
)

= Tried and not kept

- *Packed counter readbacks (F8).* One `Int32` buffer and one device-to-host copy per round. 1.04–1.11× in 2D, neutral in 3D, not stable over 80 pairs. Deferred.
- *Thread per slot* for classification and for the MQS kernel. No gain for classification; at most 25% in 3D Float32 and slower in 2D for MQS.
- *Four cached nodes on vector grids.* About 6% slower than caching only the parent bounds in 3D Float64.
- *Removing launch syncs (M1).* Not pursued. Profiling showed the GPU busy for 42.3 ms of a 43.4 ms call, so launch overhead was not the bottleneck.

= Correctness

- Both versions start from identical seeded particle sets: the same initial count in every case, from 1,560,600 to 65,548,256 particles.
- Survivor counts after `move_particles!` differ by at most 82 particles (2D n=1024 Float32, 25.1 million particles). These are drops from full destination cells. The pull scheme and the colored sweeps choose different survivors in overflowing cells. The tests check slot sets, not exact slots.
- F9 is bit-identical to F7 in occupancy, slot placement, coordinates and carried fields, NaNs included.
- New regression tests: `check_full_destination_chain` (2D and 3D) for F7, and a device test that compares cached lookup with bisection for F9.
- Open: the F9 CPU and CUDA movement gates were still running in `GPU_PROFILING_PLAN.md`. AMDGPU and Metal were not tested; no such hardware on this host.

= Method

- NVIDIA GeForce RTX 5070 Ti (16 GB), driver 617.42, Windows (WDDM). Julia 1.13.3, CUDA.jl 6.5.0, KernelAbstractions 0.10.4.
- Setup follows the temperature-advection scripts: staggered velocity grids, rotational flow, seed 391, `nxcell/max_xcell` of 24/30 in 2D and 25/50 in 3D, `dt` at half the CFL limit.
- Advection timings start from the initial particles. Move timings start from one MQS-advected snapshot. The snapshot is restored outside the timer before each sample. Three warm-up calls, then 30 timed samples with `CUDA.@sync`; reported value is the median.
- Each version runs in its own environment with the same package versions, and only the JustPIC checkout differs.
- Per-change numbers come from a second same-GPU session with two interleaved runs per build: main with only F1 applied (advection), the branch's colored sweeps (F5), and the committed F7 mover at `ac50925` (F6+F7). The F9 pairs come from the profiling driver in `GPU_PROFILING_PLAN.md`.
- 2D n=256 kernels last under 1 ms, so their gains carry more noise than the larger cases.

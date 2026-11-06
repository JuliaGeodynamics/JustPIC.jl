# JustPIC.jl JOSS manuscript

First draft prepared on 5 November 2026 against checkout
`efb7b64920031673bb228cb3f48da880cb89f22b` (package version 0.7.16).
This is a working manuscript, not a submission-ready paper.

## Files

- [paper.md](paper.md): manuscript in JOSS Markdown format.
- [paper.bib](paper.bib): five references used in the draft.
- [PLAN.md](PLAN.md): writing plan and remaining preparation work.
- [generate_figures.jl](generate_figures.jl): reproducible CPU simulations and figure generation using the existing docs environment.
- [figures/](figures/): PNG/PDF/SVG figures, plotted data, and validation metadata.
- [reference-MagmaThermoKinematics.pdf](reference-MagmaThermoKinematics.pdf): the supplied reference paper, moved from the repository root and renamed to prevent accidental overwrite.

The reference PDF remains a local file excluded by the repository's existing
`*.pdf` ignore rule. No ignore rules were changed.

The manuscript follows the current [JOSS format](https://joss.readthedocs.io/en/latest/paper.html)
and [metadata example](https://joss.readthedocs.io/en/latest/example_paper.html).
The reference PDF guides style only; no application results or figures were copied.

## Progress and author decisions

- [x] Move manuscript preparation materials into this folder.
- [x] Write the first draft with all required section headings.
- [x] Use the four authors, order, affiliations, and ORCIDs from `.zenodo.json`; the user approved this choice for the draft.
- [x] Check core design claims against implementation and documentation.
- [x] Establish a published downstream integration and historical release evidence.
- [x] Add bibliography entries from primary sources and the dependency's preferred citation.
- [ ] Confirm corresponding author, each coauthor's consent, and current metadata.
- [ ] Have the developers verify the build-versus-contribute rationale and design trade-offs.
- [ ] Confirm all funding, grant numbers, sponsor roles, and conflicts. The draft uses only the funding stated in the root README; `.zenodo.json` lists additional support requiring reconciliation.
- [ ] Complete the AI disclosure for historical code/documentation work and verify the model description. Add human review/validation assurances only after the authors perform those checks.
- [ ] Add further research uses if desired. Do not describe the JustRelax integration as independent external adoption: authors overlap.
- [ ] Choose the reviewed release and verify its archive DOI. The JustRelax bibliography links JustPIC's concept DOI, `10.5281/zenodo.10212675`, and a historical v0.5.10 archive, `10.5281/zenodo.15704602`. Neither is asserted to be the DOI for the current checkout.
- [ ] Run installation, CPU tests, and relevant backend checks for the reviewed revision; inspect the actual CI results.
- [x] Include two numerical illustrations with a regenerating script and assertions; see the figure methods below.
- [ ] Compile with the JOSS toolchain and obtain coauthor approval.

Author actions are also marked with HTML comments in the manuscript. Those comments
do not appear in the rendered paper; they must be resolved before submission.

## Evidence behind the draft

Sources accessed on 5 November 2026. Repository implementation claims refer to the
local checkout; remote `main` files may evolve.

| Claim | Evidence | Limit |
| --- | --- | --- |
| Shared backend implementation | [`src/JustPIC.jl`](../src/JustPIC.jl), [`src/common.jl`](../src/common.jl), [`CUDA extension`](../ext/JustPICCUDAExt.jl) | No performance equivalence inferred. |
| Cell-local storage and capacity | [`CellArrays guide`](../docs/src/CellArrays.md), [`particle lifecycle`](../docs/src/particle_lifecycle.md), [`move_safe.jl`](../src/Particles/move_safe.jl) | Capacity and non-periodic escape may remove particles. |
| Nodal gather avoids competing particle writes | [`particle_to_grid.jl`](../src/Interpolations/particle_to_grid.jl), [`interpolation guide`](../docs/src/interpolations.md) | Specific to this transfer; not a blanket claim that the package uses no atomics. |
| Precision and distributed support | [`support matrix`](../docs/src/support_matrix.md) | Metal lacks Float64; material-particle MPI is experimental; multi-rank passive markers are unsupported. |
| Published research integration | [JustRelax JOSS article](https://joss.theoj.org/papers/10.21105/joss.09365), [paper source](https://github.com/PTsolvers/JustRelax.jl/blob/main/docs/paper/paper.md) | Explicitly identifies JustPIC; no new scientific results claimed here. |
| Inspectable downstream use | [JustRelax README](https://github.com/PTsolvers/JustRelax.jl/blob/main/README.md), [shear-band example](https://ptsolvers.github.io/JustRelax.jl/dev/man/ShearBand2D), [`downstream workflow`](../.github/workflows/Downstream.yml) | Existence checked; execution was not checked. |
| Development over multiple years | [v0.2.2, 28 February 2024](https://github.com/JuliaGeodynamics/JustPIC.jl/releases/tag/v0.2.2), [v0.7.16, 2 November 2026](https://github.com/JuliaGeodynamics/JustPIC.jl/releases/tag/v0.7.16) | Release records support history; they do not independently prove continuous public visibility or every editorial gate. |
| Research role of alternatives | [Underworld2 paper](https://joss.theoj.org/papers/10.21105/joss.01797), [ASPECT paper](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1365-246x.2012.05609.x), [ASPECT particle cookbook](https://aspect-documentation.readthedocs.io/en/latest/user/cookbooks/cookbooks-particles.html) | No hardware ranking or speed comparison. |
| Julia citation metadata | [SIAM article](https://epubs.siam.org/doi/10.1137/141000671), [JustRelax bibliography](https://github.com/PTsolvers/JustRelax.jl/blob/main/docs/paper/paper.bib) | Bibliographic metadata checked; not a review of the language paper. |
| KernelAbstractions citation | [preferred citation](https://github.com/JuliaGPU/KernelAbstractions.jl/blob/main/CITATION.cff) | Preferred citation specifies no year; none invented. Zenodo resolution was unavailable during preparation. |

## Figure methods and regeneration

Run from the repository root using the existing docs environment:

```sh
julia --startup-file=no --project=docs paper/generate_figures.jl
```

If an older local docs manifest reports missing JustPIC dependencies, refresh it
with `julia --startup-file=no --project=docs -e 'using Pkg; Pkg.resolve()'`.

The script uses the local JustPIC checkout resolved by `docs/Manifest.toml` and
the existing CairoMakie, GLMakie, and JLD2 dependencies. GLMakie needs a working
OpenGL display/context for the 3D render. It adds no packages or changes to
the numerical library. Regular particle lattices make the runs deterministic.
The plotted runs use Float64 on the CPU, with smaller Float32 cases checked first.

**2D reversible vortex:** on the unit square,
`u = sin(πx)² sin(2πy) cos(πt/3)` and
`v = -sin(πy)² sin(2πx) cos(πt/3)`.
The initial scalar is `exp(-((r - 0.14)/0.025)²)`, where
`r = hypot(x - 0.5, y - 0.72)`. Grid values are interpolated onto particles once;
particle values then remain attached to trajectories. The figure shows nodal
reconstructions at times 0, 1.5, and 3 using one fixed colour scale. The initial
panel is a particle reconstruction too, allowing a consistent return comparison.
The 160² grid starts with sixteen particles per cell and reserves 64 slots per cell.
There are 960 RK4 steps, with time amplitude sampled at each step midpoint.
No injection is performed. Assertions require unchanged particle count, finite
physical grid values, maximum trajectory return error below 0.002 unit lengths,
and relative grid L2 return error below 0.02.

**3D reversible deformation:** select regular-lattice particles satisfying
`(hypot(x - 0.5, y - 0.5) - 0.25)² + (z - 0.5)² < 0.065²`.
Use the divergence-free field
`u = 2sin(πx)² sin(2πy) sin(2πz)`,
`v = -sin(πy)² sin(2πx) sin(2πz)`, and
`w = -sin(πz)² sin(2πx) sin(2πy)`, all multiplied by `cos(πt/2)`.
This field has zero normal velocity on the cube boundaries. The cloud stretches
and folds until `t = 1`, then retraces its trajectories as the velocity reverses,
recovering at `t = 2`. Colour follows each particle's initial azimuth.
The grid has 48³ cells, initially eight particle candidates per cell before
selecting the torus, and 64 slots per cell. The run uses 480 RK4 steps with
midpoint-sampled amplitude and no injection. Assertions require unchanged
particle count, finite snapshots inside the unit cube, and maximum particle
return error below 0.002 unit lengths. All views use the same camera and
coordinate limits of 0.1 to 0.9 on each axis. A 16³ Float32 case runs 240 steps
with the same checks before the full run.

Outputs are `vortex_transport.png/.pdf/.svg`, `deformation_3d.png/.pdf/.svg`,
`figure_data.jld2`, and `validation.toml` under `figures/`. PNGs are rendered at
3600 × 1440 pixels (vortex) and 3600 × 1680 pixels (3D deformation). The vortex PDF and SVG
keep vector contours, streamlines, axes, and text over a raster scalar field. The 3D PDF and SVG
embed the high-resolution GL render so its lighting is preserved. The JLD2 file stores the
actual plotted snapshots. TOML metadata records sizes, steps, counts, errors,
versions, source path, revision, and script/manifest hashes. These are
illustrative checks, not a convergence study or a performance benchmark.

To restyle the saved snapshots without rerunning either simulation:

```sh
julia --startup-file=no --project=docs paper/generate_figures.jl --render-only
```

The 3D cloud uses actual particle positions rendered as opaque spheres of radius
0.0055 unit lengths, with warm key and cool fill lights, specular highlights, screen-space ambient occlusion
(SSAO), anti-aliasing (FXAA), and translucent axis panels. These are visual depth
cues, not simulated physical illumination or ray tracing. The cyclic `phase`
colour map tracks initial azimuth across all panels; lighting modulates its
apparent brightness. Particle sizes are illustrative, not physical radii.
The vortex retains a fixed `batlow` scalar scale with white contours at 0.25 and
0.65. No smoothing or fitted surface is introduced.
Both figures overlay streamlines computed from the same spatial velocity helpers
used by their simulations. Faint curves show the time-independent spatial flow
pattern, while arrows show the initial or reversed direction. At the midpoint,
the temporal amplitude vanishes: the reference curves remain visible without
arrowheads and do not represent nonzero instantaneous flow. The 3D curves are
translucent and depth-tested against the opaque particles.

The earlier `torus_rotation` exports are retained for comparison; the manuscript
and regeneration script now use `deformation_3d`.

`--render-only` preserves `figure_data.jld2` and the original numerical
`validation.toml`. A separate `rendering.toml` records the rendering script and
cached-data hashes, renderer versions, and export method for either command.

The figure folder locally overrides the repository's image/PDF/data ignore
patterns so the new manuscript assets can be committed. The reference PDF
remains excluded. GPU execution and equality between backends are not tested by
these CPU runs.

## Validation and rendering

Run the repository-independent Markdown scanner from the installed reviewer-two
skill, if available:

```powershell
node C:/Users/Albert/.codex/plugins/cache/reviewer-two/reviewer-two/0.2.0/skills/manuscript-conventions/scripts/mdscan.js paper/paper.md
```

For an official JOSS preview, run the following **from the repository root** on a
machine with Docker. The mounted folder contains all manuscript inputs and
receives the generated `paper.pdf`; the reference PDF has a different name.

```powershell
$manuscriptDir = (Resolve-Path ./paper).Path
docker run --rm --volume "${manuscriptDir}:/data" --env JOURNAL=joss openjournals/inara
```

This follows the [JOSS build instructions](https://joss.readthedocs.io/en/latest/paper.html#checking-that-your-paper-compiles).
The Markdown source scan found no citation, label, or missing-image issues.
All local links and required section headings were checked; the draft is within
the 750–1750-word range.
The figure-generation script has been run on the CPU, including its small
Float32 checks; PNG and PDF figure layouts were visually inspected. This does
not replace the package test suite or backend checks. The manuscript itself has
not been compiled with the JOSS toolchain; Docker, Pandoc, and Quarto were absent
from PATH during preparation. Its final PDF layout, font embedding, and rendered
treatment of the undated software citation still need checking.

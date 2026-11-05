# Plan for a short JustPIC.jl JOSS paper

Prepared: 5 November 2026. Journal guidance checked on this date.

## Aim and scope

Write a concise software paper explaining why JustPIC.jl is useful to geodynamics researchers, what distinguishes its design, and where it is already used. Proposed title: **JustPIC.jl: Particle-in-cell advection and particle–grid coupling on CPUs and GPUs**.

The central story is a reusable Julia library that separates particle transport and field transfer from the host simulation, with shared implementations for 2D/3D CPU and GPU workflows. Explain particle-in-cell in ordinary language before introducing specialist terms. Do not imply that JustPIC solves the full geodynamic equations itself.

Work started on 5 November 2026. The manuscript and bibliography are in this folder; see README.md for progress, checked sources, and remaining author decisions.

## Verified journal requirements

- Submit Markdown with YAML metadata and bibliography; keep the manuscript alongside the software in its Git repository. The repository must be openly accessible and licensed. Public development must exceed six months, with sustained iteration and demonstrated research use. Prepare evidence before submission. [Submission guidelines](https://joss.readthedocs.io/en/latest/submitting.html).
- JOSS specifies **750–1750 words**. Use the section names in the outline below and keep API instructions in the manual. Supply author/affiliation metadata, scholarly references, and funding acknowledgements. [Paper format](https://joss.readthedocs.io/en/latest/paper.html).
- Review covers installation, functionality, documentation, tests, community guidance, and research significance. Comparisons must justify building this package rather than contributing the same functionality elsewhere. [Review criteria](https://joss.readthedocs.io/en/latest/review_criteria.html).
- Disclose AI tools/models, versions, scope, and human verification. Include this planning assistance if it informs the paper; establish the broader development history with the authors. AI-generated author–editor/reviewer conversation is prohibited except translation. [AI policy](https://joss.readthedocs.io/en/latest/submitting.html#ai-usage-policy).
- After successful review, create the reviewed release and archive it for a software DOI. Disclose conflicts, funding, and sponsor involvement. [Submission and review process](https://joss.readthedocs.io/en/latest/submitting.html).

These are current requirements, not guarantees of editorial acceptance. Recheck the linked guidance immediately before submission.

## How to use the reference PDF

The local [reference PDF](reference-MagmaThermoKinematics.pdf), originally `paper.pdf` in the repository root, is a **draft about MagmaThermoKinematics.jl**, with placeholder publication metadata. Use its short paragraphs, accessible opening, ecosystem context, and design-to-application narrative as a stylistic reference. It already illustrates sections such as Software design, Research impact statement, and AI usage disclosure.

Do not copy its application results, numerical values, author metadata, references, or AI statement into the JustPIC paper. Its Physics and Example Applications sections are optional inspiration: for this smaller paper, integrate the necessary material into Software design and Research impact statement. Use JOSS's current Markdown example for source formatting; a PDF is not an editable submission template. Never overwrite the supplied PDF when rendering the new manuscript.

## Evidence already available in the checkout

| Material | Use in the paper or preparation | Qualification |
| --- | --- | --- |
| [README.md](../README.md), [Project.toml](../Project.toml) | Purpose, installation, dependencies, capabilities | Checkout version is 0.7.16; select the reviewed version later. |
| [src/JustPIC.jl](../src/JustPIC.jl), [src/common.jl](../src/common.jl), `ext/` | Verify shared implementation and backend extension boundaries | Read the relevant implementation before making architectural claims. |
| [Particle lifecycle](../docs/src/particle_lifecycle.md), [CellArrays](../docs/src/CellArrays.md), [Interpolations](../docs/src/interpolations.md) | Explain storage, movement, and particle–grid coupling | Describe why choices matter, rather than listing functions. |
| [Support matrix](../docs/src/support_matrix.md), [Backends](../docs/src/backends.md) | Bound hardware, dimensionality, and precision claims | Metal lacks Float64; support varies by subsystem. |
| [test/runtests.jl](../test/runtests.jl), `.github/workflows/`, `.buildkite/` | Establish correctness and reproducibility evidence | Configuration exists; successful runs were not verified for this plan. |
| [Benchmarking guide](../benchmarking/README.md), [Performance](../docs/src/performance.md) | Reuse existing timing and provenance machinery if needed | No performance numbers are endorsed here. |
| [Downstream workflow](../.github/workflows/Downstream.yml) | Starting evidence for integration with JustRelax.jl | Obtain a concrete downstream example and research citation. |
| [LICENSE](../LICENSE), [CONTRIBUTING.md](../CONTRIBUTING.md) | MIT licence and community pathways | Check linked contribution/support routes before submission. |
| [AUTHORS.md](../AUTHORS.md), [.zenodo.json](../.zenodo.json) | Starting points for authorship, ORCIDs, affiliations, funding | Confirm with contributors; existing metadata is not final paper metadata. |

## Proposed outline and word budget

Target approximately **1350 words of prose**, leaving room within the journal range. Budgets below are writing targets, not separate journal limits; check captions and the final rendered manuscript as well.

| Section | Words | JustPIC-specific content |
| --- | ---: | --- |
| Summary | 150 | What particles carry, how they exchange information with a grid, research audience, 2D/3D support, and CPU/GPU portability. |
| Statement of need | 180 | Why transporting material properties in geodynamic simulations needs reusable advection, interpolation, and occupancy management. Explain the burden this removes from host solvers. |
| State of the field | 220 | Compare two or three relevant alternatives and distinguish reusable particle components from complete simulation codes. Give the authors' concrete reason for a separate library. |
| Software design | 350 | Cell-local storage; advection, movement and injection; grid/particle transfers; shared KernelAbstractions kernels and optional GPU extensions. Explain trade-offs and limitations. |
| Research impact statement | 250 | Verified research uses and integrations, with citations and reproducible examples. Explain the specific role JustPIC played. |
| AI usage disclosure | 100 | Verified tool/model history, assistance scope, and actual human checks. |
| Acknowledgements | 50 | Confirmed funding, grant identifiers where applicable, contributions, and sponsor role. |
| References | — | Focused bibliography, generated from `paper.bib`. |

In Software design, ask the developers to explain the locality-versus-capacity trade-off of cell-local storage and the maintainability-versus-backend-tuning trade-off of shared kernels. Confirm these rationales rather than inventing design history. Mention phase ratios, passive markers, free-surface markers, and checkpointing briefly when they support the main story.

Avoid blanket claims of conservation, unrestricted particle capacity, identical results on every backend, or universal MPI support. In particular, read `src/Particles/move_safe.jl` before describing movement guarantees. The support matrix labels material-particle MPI experimental and multi-rank passive markers unsupported.

## Work sequence

### 1. Resolve author inputs and submission readiness

- [ ] Agree author list/order, corresponding author, current affiliations, ORCIDs, contributions, and consent. `AUTHORS.md` names Albert de Montserrat, Pascal Aellig, Thibault Duretz, and Ludovic Räss; this is a starting list for discussion.
- [ ] Confirm funding and conflicts. Reconcile `.zenodo.json` with current documentation: its description omits Metal and mentions matrix-free solvers, so do not reuse it uncritically.
- [x] Collect historical release and associated issue/PR links: v0.2.2 (February 2024) and v0.7.16 (November 2026), recorded in README.md. Continuous public visibility and the full editorial history criteria remain to verify.
- [x] Verify JustRelax as a research integration through its published JOSS paper and documented examples. Add independent adopters or further applications if available; reviewed software version remains to select.
- [ ] Record AI assistance across code, documentation, and manuscript work, including this Codex-assisted plan. Verify model/version details where available; do not invent them.

**Completion:** a short evidence list and confirmed author metadata; unresolved facts remain explicit author queries.

### 2. Establish the comparison and bibliography

- [x] Identify Underworld2 and ASPECT through primary papers and documentation; distinguish their modelling-framework scope from JustPIC's particle component.
- [ ] Compare research role, particle/grid functionality, integration model, and documented hardware support. Avoid unsupported superiority or novelty claims.
- [ ] Explain why reusing dependencies while exposing a separate particle library suited the intended applications better than extending an existing alternative.
- [x] Add five references covering Julia, KernelAbstractions, Underworld2, ASPECT, and JustRelax. Add method-specific citations if needed and the reviewed software archive once verified.

**Completion:** a defensible comparison paragraph and a small, checked bibliography.

### 3. Select one illustration and only necessary measurements

- [ ] Prefer one compact figure from an existing example: a particle-carried field before/after advection and its reconstruction on the grid. Start with `docs/examples/field_advection2D.jl` and its manual page; confirm inputs and expected behavior.
- [ ] Save a regenerating script, environment/version information, and caption describing what the image demonstrates. Include a 3D panel only if it adds essential information.
- [ ] If numerical accuracy or speed is claimed, identify the supporting test or run. Use a defined error metric/reference solution for accuracy.
- [ ] If timing is useful, reuse the benchmark runner. Record revision, Julia/package versions, device, precision, sizes, synchronization, warm-up, and dispersion. Compare equivalent workloads; distinguish modeled bandwidth/FLOPs from hardware measurements.

**Completion:** one reproducible illustrative result. Omit speedups and scaling claims unless measured; a new benchmark campaign is not needed merely to fill the paper.

### 4. Draft and render

- [x] Create `paper.md` and `paper.bib` in this folder. Preserve the supplied PDF as `reference-MagmaThermoKinematics.pdf`; the optional figure is deferred.
- [x] Follow the official JOSS metadata example. Populate YAML title, authors, affiliation indices, ORCIDs, tags, date, and bibliography; keep the title plain text. Author metadata approved for this draft.
- [x] Write a first draft using the required outline, checked implementation claims, and a published research integration. Author confirmation comments remain before submission.
- [ ] Compile with the documented Open Journals/Inara workflow and inspect metadata, citations, figure labels, text overflow, and length. Keep build output under `paper/` or a temporary directory.
- [ ] Obtain coauthor review of scientific claims, attribution, design rationale, and AI disclosure.

**Completion:** a readable, compiled short manuscript approved by all authors.

### 5. Verify the software and prepare submission

- [ ] Ask a colleague to follow installation and the chosen example from a clean environment.
- [ ] Run the documented CPU suite: `julia --project=. -e 'using Pkg; Pkg.test()'`. Record the revision and result; inspect existing GPU CI or run relevant backend suites for hardware claims.
- [ ] Check documentation and community links. Build the manual if documentation changes are made. Confirm the public paper branch contains the software under review.
- [ ] Complete the journal checklist, recheck policies, and have the corresponding author submit. Human authors handle editor/reviewer conversation under the current AI policy.
- [ ] After review changes, tag the reviewed release, archive that exact version, obtain its DOI, and reconcile paper and archive metadata before acceptance.

**Stop condition:** the short paper, checked bibliography, reproducible illustration if used, evidence of research use, and reviewable software are ready. Do not expand this into a methods monograph, API tutorial, new feature project, or comprehensive hardware benchmark study.

## Not established by this plan

No tests, examples, benchmarks, or manuscript compilation were run. The manuscript source scan found no citation or cross-reference issues; all local Markdown links and required headings were checked. The first draft was approximately 1080 words before final copy edits. Published JustRelax integration and historical releases were verified, but full public-history eligibility, final coauthor consent/funding, archive DOI, and historical AI use still need confirmation. Existing files and CI configuration provide preparation evidence, not proof that the current checkout passes review.

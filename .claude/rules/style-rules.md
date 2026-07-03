---
paths:
  - src/**/*.jl
  - ext/**/*.jl
  - test/**/*.jl
  - docs/**/*.jl
---

# Style Rules

## Formatting: Runic

This repo is formatted with [Runic](https://github.com/fredrikekre/Runic.jl), **not**
JuliaFormatter. Before pushing:

```sh
git runic main
```

CI (`Format.yml`) posts a diff comment on PRs that need formatting. Don't hand-format
against Runic's output; let the tool decide spacing/indentation questions.

## Naming

- **Types**: PascalCase — `Particles`, `MarkerChain`, `RungeKutta2`
- **Functions**: snake_case, `!` for mutation — `advection!`, `inject_particles!`
- **Kernels**: the `@kernel` function takes the `_kernel!` suffix
  (`backtrack_kernel!`); the launching wrapper keeps the plain name
- **Established grid shorthand** (use these, don't invent synonyms):
  `xci` cell centers, `xvi` cell vertices, `di` grid spacing, `_di` inverse spacing,
  `ni` grid size, `nxcell`/`min_xcell`/`max_xcell` cell occupancy, `ip`/`ipart` particle
  slot index, `I...` cell index
- Unicode math is welcome where the codebase already uses it (`pᵢ`, `Δt`, `α`); pick one
  notation per scope and stay consistent

## Comments

Default to **no comment** — clear names and structure over prose. Add a comment only for a
non-obvious invariant, a numerical-stability detail, a sign convention that contradicts
intuition, or a workaround for an upstream bug — one line, at exactly the confusing step
(see the header of [src/launch.jl](../../src/launch.jl) for the house style on
module-level notes).

- ❌ `# loop over particles` above a loop
- ✅ `# 0 block length == struct-of-array layout, required on GPU`
- Never narrate the change ("added for X", "fixes review comment") — that belongs in the
  commit message
- Delete commented-out code; git is the journal

## General

- Always use explicit `return` in multi-expression functions
- Keep PRs focused: unrelated cleanup goes in a separate PR
- Update exports in `src/common.jl` (or `src/JustPIC.jl` for struct-level API) when adding
  public functions; docs API page (`docs/src/API.md`) follows exports

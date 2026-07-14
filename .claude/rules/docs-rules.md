---
paths:
  - docs/**/*
---

# Documentation Rules

## Building Docs

```sh
julia --project=docs -e 'using Pkg; Pkg.instantiate()'   # once
julia --project=docs docs/make.jl
```

`docs/make.jl` pushes the repo root onto `LOAD_PATH` — the in-repo JustPIC is used
directly, no `Pkg.develop` needed.

## Fast Local Builds

For prose/cross-reference checks only, temporarily set `warnonly = true` in `makedocs`
and/or comment out the heavier `"Examples"` pages. **Revert before committing.**

## Viewing

```julia
using LiveServer
serve(dir = "docs/build")
```

## Style

- Doc sources are plain markdown in `docs/src/`; example scripts in `docs/examples/`
- `docs/src/API.md` is the public API page — keep it in sync with exports in
  `src/common.jl` / `src/JustPIC.jl`
- Prefer `jldoctest` blocks when output is stable and backend-independent; GPU array
  output cannot be doctested — use plain `julia` blocks there
- Unicode math (`Δt`, `xci`, `ρ`), not LaTeX, in docstrings

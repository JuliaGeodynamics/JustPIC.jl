# Documentation

## Building Docs Locally

```sh
julia --project=docs docs/make.jl
```

`docs/make.jl` pushes the repo root onto `LOAD_PATH`, so the in-repo JustPIC is used
directly — no `Pkg.develop` needed. Instantiate once with
`julia --project=docs -e 'using Pkg; Pkg.instantiate()'`.

## Fast Local Builds

For prose/cross-reference checks, temporarily:
- extend `warnonly` (e.g. `warnonly = true`) in `makedocs`
- comment out the heavier example pages under `"Examples"` in `pages`

**Revert these changes before committing.**

## Viewing Docs

```julia
using LiveServer
serve(dir = "docs/build")
```

## Structure

- Doc sources in `docs/src/*.md`; example walk-throughs are plain markdown
  (`field_advection2D.md` etc.) with fenced julia blocks
- The public API page is `docs/src/API.md` — update it when exports change
- Example scripts referenced by the docs live in `docs/examples/`

## Docstring Style

- Docstrings on exported functions: signature line, one-sentence description, then
  arguments; see `launch!`/`cell_array` in [src/launch.jl](../src/launch.jl) for the
  house style
- Prefer `jldoctest` blocks over plain `julia` blocks when the output is stable and
  backend-independent — doctests are verified by the docs build, plain blocks rot.
  Backend-dependent output (GPU array types) cannot be doctested; use plain blocks there
- Use unicode math (`Δt`, `xci`, `ρ`) as in the codebase, not LaTeX

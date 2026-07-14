---
name: build-docs
description: Build JustPIC documentation locally, full or fast
---

# Build Documentation

## Steps

1. Ask whether the user wants a **full build** or a **fast build** (prose/cross-reference
   check only)
2. Instantiate the docs environment if needed:
   ```sh
   julia --project=docs -e 'using Pkg; Pkg.instantiate()'
   ```
3. **Full build**:
   ```sh
   julia --project=docs docs/make.jl
   ```
4. **Fast build** — temporarily edit `docs/make.jl`:
   - set `warnonly = true` in `makedocs`
   - optionally comment out the `"Examples"` pages
   - run the build, then **revert all `docs/make.jl` changes**
5. Preview:
   ```julia
   using LiveServer
   serve(dir = "docs/build")
   ```
6. Report warnings/errors, especially missing docstrings and broken cross-references

## Notes

- `docs/make.jl` puts the repo root on `LOAD_PATH` — the local source is used, no dev/add
- If exports changed, check `docs/src/API.md` is still complete before building

# Indexing CellArrays

Objects of the `CellArray` type cannot be directly indexed or mutated at the particle level. We can however use the `@cell` macro to index and mutate the values of a `CellArray` in a friendly way. Let's start by initializing a `CellArray` with $10 \times 10$ cells, where every cell contains 4 cells. We do this with help from ParallelStencil.jl

````julia
using CellArrays, ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

A = @zeros(10, 10, celldims = (4,))
````

We can read the, for example, cell (1,1) as
```julia-repl
julia> A[1, 1]
4-element StaticArraysCore.SVector{4, Float64} with indices SOneTo(4):
 0.0
 0.0
 0.0
 0.0
```

however, cell values cannot be directly modified as they are a `StaticVector`. For this we need help of the `@cell` macro. For example, if we want to set the value of the second particle of the cell (1, 1) to 1.0 we can do it like this:

````julia
@cell A[2, 1, 1] = 1.0
````
and now

```julia-repl
julia> A[1, 1]
4-element StaticArraysCore.SVector{4, Float64} with indices SOneTo(4):
 0.0
 1.0
 0.0
 0.0
```

Finally, if we want read-only to the i-th particles within the cell (1, 1) we can use the `@cell` macro like this:

````julia
@cell A[2, 1, 1]
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*


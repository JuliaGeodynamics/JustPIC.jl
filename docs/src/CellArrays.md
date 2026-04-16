# Working with CellArrays

`CellArray`s are the storage primitive behind JustPIC particle containers. They represent a grid where every logical grid cell stores a small fixed-size payload, for example the particle slots belonging to that cell.

## Instantiating a `CellArray`

With `ParallelStencil.jl`, `CellArray`s can be created with the familiar
allocation macros. The `celldims` keyword controls the payload size inside each
grid cell.

```julia
using JustPIC, JustPIC._2D
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)
```

```julia-repl
julia> ni = (2, 2)
(2, 2)

julia> ncells = (2,)
(2,)

julia> x = 20
20

julia> CA = @fill(x, ni..., celldims = ncells, eltype = Float64) 
2×2 CellArrays.CPUCellArray{StaticArraysCore.SVector{2, Float64}, 2, 1, Float64}:
 [20.0, 20.0]  [20.0, 20.0]
 [20.0, 20.0]  [20.0, 20.0]
```

## Indexing a `CellArray`

Indexing by grid cell returns the whole payload stored in that cell. This is
convenient for inspection, but it materializes a `StaticArray` value:

```julia-repl 
julia> CA[1,1]
2-element StaticArraysCore.SVector{2, Float64} with indices SOneTo(2):
 20.0
 20.0
```

Performance-sensitive kernels usually read or mutate individual payload entries
directly. For this purpose, JustPIC exports `@index`.

For example, to read a single element of `CA`:

```julia-repl
julia> @index CA[2, 1, 1]
20.0
```

Here the first index selects the payload entry and the remaining indices select
the grid cell. Mutation uses the same syntax:

```julia-repl
julia> @index CA[2, 1, 1] = 0.0
0.0

julia> CA
2×2 CellArrays.CPUCellArray{StaticArraysCore.SVector{2, Float64}, 2, 1, Float64}:
 [20.0, 0.0]   [20.0, 20.0]
 [20.0, 20.0]  [20.0, 20.0]
```

`@cell` is the companion macro for reading or writing an entire cell payload:

```julia-repl 
julia> @cell CA[1,1]
2-element StaticArraysCore.SVector{2, Float64} with indices SOneTo(2):
 20.0
 20.0
```

```julia-repl 
julia> @cell CA[1,1] = @cell(CA[1,1]) .+ 1
2-element StaticArraysCore.SVector{2, Float64} with indices SOneTo(2):
 21.0
 21.0

 julia> CA
2×2 CellArrays.CPUCellArray{StaticArraysCore.SVector{2, Float64}, 2, 1, Float64}:
 [21.0, 21.0]  [20.0, 20.0]
 [20.0, 20.0]  [20.0, 20.0]
```

## Helper Functions

The most common low-level helpers are:

- `cellnum(A)`: number of payload entries stored in each logical cell.
- `cellaxes(A)`: one-based axes for iterating over payload entries.
- `cell_array(x, ncells, ni)`: allocate a cell array filled with `x`.

Most users interact with these indirectly through `init_particles`,
`init_cell_arrays`, and the interpolation/advection kernels.

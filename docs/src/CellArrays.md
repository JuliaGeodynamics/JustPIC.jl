# Working with CellArrays

## Instantiating a `CellArray` 

With the help of `ParallelStencil.jl` we can easily create a `CellArray` object. The `CellArray` object is a container that holds the data of a grid. The data is stored in small nD-arrays, and the grid is divided into cells. Each cell contains a number of elements. The `CellArray` object is used to store the data of the particles in the simulation.

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

We can access to the data of one `CellArray` by indexing a given grid cell. This will however instantiate a `StaticArray` object with the data of the cell. 

```julia-repl 
julia> CA[1,1]
2-element StaticArraysCore.SVector{2, Float64} with indices SOneTo(2):
 20.0
 20.0
```

It is however useful to read and mutate the data of the `CellArray` object directly, without instantiating a `StaticArray`. For this porpuse, `JustPIC` exports the macro `@index` to directly read and mutate the individual elements of the cell. 

For example, to read a single element of `CA`:

```julia-repl
julia> @index CA[2, 1, 1]
20.0
```

where, in this case, the first index corresponds to the 2nd element of the data within $cell_{11}$ cell. We can mutate the `CellArray` in a similar way:

```julia-repl
julia> @index CA[2, 1, 1] = 0.0
0.0

julia> CA
2×2 CellArrays.CPUCellArray{StaticArraysCore.SVector{2, Float64}, 2, 1, Float64}:
 [20.0, 0.0]   [20.0, 20.0]
 [20.0, 20.0]  [20.0, 20.0]
```

`JustPIC` also provides the macro `@cell` operatig at the cell level:

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


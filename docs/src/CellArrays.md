# Working with CellArrays

`CellArray`s are the storage primitive behind JustPIC particle containers. They represent a grid where every logical grid cell stores a small fixed-size payload, for example the particle slots belonging to that cell.

## Instantiating a `CellArray`

`cell_array(backend, value, ncells, ni)` allocates a `CellArray` over an `ni`-shaped
grid, giving every cell an `ncells`-shaped payload initialized to `value`. The payload
is what holds per-particle data — coordinates, temperature, phase labels — for the
particles belonging to that grid cell.

```jldoctest cellarrays
julia> using JustPIC

julia> import CellArraysIndexing as CAI

julia> ni = (2, 2)      # grid cells
(2, 2)

julia> ncells = (2,)    # payload slots per cell
(2,)

julia> CA = cell_array(JustPIC.CPU, 20.0, ncells, ni)
2×2 CellArrays.CPUCellArray{StaticArraysCore.SVector{2, Float64}, 2, 1, Float64}:
 [20.0, 20.0]  [20.0, 20.0]
 [20.0, 20.0]  [20.0, 20.0]
```

## Indexing a `CellArray`

Indexing by grid cell returns the whole payload stored in that cell. This is
convenient for inspection, but it materializes a `StaticArray` value:

```jldoctest cellarrays
julia> CA[1, 1]
2-element StaticArraysCore.SVector{2, Float64} with indices SOneTo(2):
 20.0
 20.0
```

It is however useful to read and mutate the data of the `CellArray` object directly, without instantiating a `StaticArray`. For this purpose, `CellArraysIndexing` provides `@index` to directly read and mutate the individual elements of the cell.

For example, to read a single element of `CA`:

```jldoctest cellarrays
julia> CAI.@index CA[2, 1, 1]
20.0
```

Here the first index selects the payload entry and the remaining indices select
the grid cell. Mutation uses the same syntax:

```jldoctest cellarrays
julia> CAI.@index CA[2, 1, 1] = 0.0;

julia> CA
2×2 CellArrays.CPUCellArray{StaticArraysCore.SVector{2, Float64}, 2, 1, Float64}:
 [20.0, 0.0]   [20.0, 20.0]
 [20.0, 20.0]  [20.0, 20.0]
```

`@cell` is the companion macro for reading or writing an entire cell payload:

```jldoctest cellarrays
julia> @cell CA[1, 1]
2-element StaticArraysCore.SVector{2, Float64} with indices SOneTo(2):
 20.0
  0.0

julia> @cell CA[1, 1] = @cell(CA[1, 1]) .+ 1;

julia> CA
2×2 CellArrays.CPUCellArray{StaticArraysCore.SVector{2, Float64}, 2, 1, Float64}:
 [21.0, 1.0]   [20.0, 20.0]
 [20.0, 20.0]  [20.0, 20.0]
```

## Backend and layout notes

`cell_array` dispatches on the KernelAbstractions backend, so the same call allocates a
device-resident `CellArray` when a vendor extension is loaded. The backing layout differs
between them: the CPU path uses a block length of 1, while GPU backends use a
structure-of-arrays layout. Always go through `CAI.@index`, `@cell`, `cellaxes` and
`cellnum` rather than indexing `CA.data` directly, so that code stays portable across
backends.

## API

```@docs
cell_array
CA
cellaxes
cellnum
update_cell_halo!
```

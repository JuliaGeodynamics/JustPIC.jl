module _2D
    using ImplicitGlobalGrid
    # using MPI: MPI
    using MuladdMacro
    using KernelAbstractions
    # Bare `@index` must resolve to KernelAbstractions' (needed inside `@kernel`).
    # Both CellArraysIndexing and (via `using ..JustPIC`) the parent module also
    # export `@index`; this explicit import wins the conflict. CellArraysIndexing's
    # element access is used through the `CAI.@index` qualifier instead.
    import KernelAbstractions: @index
    using CellArrays, StaticArrays
    using CellArraysIndexing: @cell, getcell, setcell!, getcellindex, setcellindex!
    import CellArraysIndexing as CAI
    using ..JustPIC
    using GridGeometryUtils

    import ..JustPIC: AbstractBackend, CPUBackend

    export CA

    CA(::Type{CPUBackend}, dims; eltype = Float64) = CPUCellArray{eltype}(undef, dims)

    include("common.jl")
end

module _3D
    using ImplicitGlobalGrid
    # using MPI: MPI
    using MuladdMacro
    using KernelAbstractions
    # Bare `@index` must resolve to KernelAbstractions' (needed inside `@kernel`).
    # Both CellArraysIndexing and (via `using ..JustPIC`) the parent module also
    # export `@index`; this explicit import wins the conflict. CellArraysIndexing's
    # element access is used through the `CAI.@index` qualifier instead.
    import KernelAbstractions: @index
    using CellArrays, StaticArrays
    using CellArraysIndexing: @cell, getcell, setcell!, getcellindex, setcellindex!
    import CellArraysIndexing as CAI
    using GridGeometryUtils

    using ..JustPIC

    import ..JustPIC: AbstractBackend, CPUBackend

    export CA

    CA(::Type{CPUBackend}, dims; eltype = Float64) = CPUCellArray{eltype}(undef, dims)

    include("common.jl")
end

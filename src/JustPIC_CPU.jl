module _2D
    using ImplicitGlobalGrid
    using MPI: MPI
    using MuladdMacro
    using ParallelStencil
    using CellArrays
    using Atomix
    using ..JustPIC

    __precompile__(false)

    @init_parallel_stencil(Threads, Float64, 2)
    
    export CA

    CA(::Type{CPUBackend}, dims; eltype=Float64) = CPUCellArray{eltype}(undef, dims)

    macro myatomic(expr)
        esc(quote
            Atomix.@atomic :monotonic $expr
        end)
    end
    
    include("common.jl")
end

module _3D
    using ImplicitGlobalGrid
    using MPI: MPI
    using MuladdMacro
    using ParallelStencil
    using CellArrays
    using Atomix
    using ..JustPIC

    __precompile__(false)

    @init_parallel_stencil(Threads, Float64, 3)

    export CA

    CA(::Type{CPUBackend}, dims; eltype=Float64) = CPUCellArray{eltype}(undef, dims)

    macro myatomic(expr)
        esc(quote
            Atomix.@atomic :monotonic $expr
        end)
    end

    include("common.jl")
end

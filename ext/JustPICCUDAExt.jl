module JustPICCUDAExt

using CUDA
using JustPIC

JustPIC.TA(::Type{CUDABackend}) = CuArray

module _2D
    using CUDA
    using ImplicitGlobalGrid
    using MPI: MPI
    using MuladdMacro, ParallelStencil, CellArrays, CellArraysIndexing, StaticArrays
    using JustPIC

    @init_parallel_stencil(CUDA, Float64, 2)

    import JustPIC: Euler, RungeKutta2, AbstractAdvectionIntegrator
    import JustPIC._2D.CA
    import JustPIC: Particles, PassiveMarkers
    import JustPIC: AbstractBackend

    export CA

    function JustPIC._2D.CA(::Type{CUDABackend}, dims; eltype=Float64)
        return CuCellArray{eltype}(undef, dims)
    end

    macro myatomic(expr)
        return esc(
            quote
                CUDA.@atomic $expr
            end,
        )
    end

    include(joinpath(@__DIR__, "../src/common.jl"))
    include(joinpath(@__DIR__, "../src/CUDAExt/CellArrays.jl"))

    JP_module = :(JustPIC._2D)
    ext_backend = :CUDABackend
    TArray = :(CUDA.CuArray)
    include("common.jl")

end

module _3D
    using CUDA
    using ImplicitGlobalGrid
    using MPI: MPI
    using MuladdMacro, ParallelStencil, CellArrays, CellArraysIndexing, StaticArrays
    using JustPIC

    @init_parallel_stencil(CUDA, Float64, 3)

    macro myatomic(expr)
        return esc(
            quote
                CUDA.@atomic $expr
            end,
        )
    end

    import JustPIC:
        Euler, RungeKutta2, AbstractAdvectionIntegrator, Particles, PassiveMarkers
    import JustPIC: AbstractBackend

    function JustPIC._3D.CA(::Type{CUDABackend}, dims; eltype=Float64)
        return CuCellArray{eltype}(undef, dims)
    end

    include(joinpath(@__DIR__, "../src/common.jl"))
    include(joinpath(@__DIR__, "../src/CUDAExt/CellArrays.jl"))

    JP_module = :(JustPIC._3D)
    ext_backend = :CUDABackend
    TArray = :(CUDA.CuArray)
    include("common.jl")

end

end # module

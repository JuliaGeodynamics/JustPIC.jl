module JustPICCUDAExt

using CUDA
using JustPIC

JustPIC.TA(::Type{CUDABackend}) = CuArray

function CUDA.CuArray(particles::JustPIC.Particles{JustPIC.CPUBackend}) 
    (; coords, index, nxcell, max_xcell, min_xcell, np) = particles
    coords_gpu = CuArray.(coords);
    return JustPIC.Particles(CUDABackend, coords_gpu, CuArray(index), nxcell, max_xcell, min_xcell, np)
end

function CUDA.CuArray(phase_ratios::JustPIC.PhaseRatios{JustPIC.CPUBackend}) 
    (; vertex, center) = phase_ratios
    return JustPIC.PhaseRatios(CUDABackend, CuArray(vertex), CuArray(center))
end

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

    local_module = @__MODULE__
    const JP_module = JustPIC._2D
    const ext_backend = CUDABackend
    const TArray = CUDA.CuArray
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

    local_module = @__MODULE__
    const JP_module    = JustPIC._3D
    const ext_backend  = CUDABackend
    const TArray       = CUDA.CuArray
    include("common.jl")

end

end # module

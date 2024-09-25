module JustPICAMDGPUExt

using AMDGPU
using JustPIC

import JustPIC: AbstractBackend, AMDGPUBackend

JustPIC.TA(::Type{AMDGPUBackend}) = ROCArray

module _2D
    using AMDGPU
    using ImplicitGlobalGrid
    using MPI: MPI
    using MuladdMacro, ParallelStencil, CellArrays, CellArraysIndexing, StaticArrays
    using JustPIC

    @init_parallel_stencil(AMDGPU, Float64, 2)

    import JustPIC: Euler, RungeKutta2, AbstractAdvectionIntegrator
    import JustPIC._2D.CA
    import JustPIC: Particles, PassiveMarkers
    import JustPIC: AbstractBackend, AMDGPUBackend

    macro myatomic(expr)
        return esc(
            quote
                AMDGPU.@atomic :monotonic $expr
            end,
        )
    end

    function JustPIC._2D.CA(::Type{AMDGPUBackend}, dims; eltype=Float64)
        return ROCCellArray{eltype}(undef, dims)
    end

    include(joinpath(@__DIR__, "../src/common.jl"))
    include(joinpath(@__DIR__, "../src/AMDGPUExt/CellArrays.jl"))

    JP_module = :(JustPIC._2D)
    ext_backend = :JustPIC.AMDGPUBackend
    TArray = :(AMDGPUBackend.ROCArray)
    include("common.jl")

end

module _3D
    using AMDGPU
    using ImplicitGlobalGrid
    using MPI: MPI
    using MuladdMacro, ParallelStencil, CellArrays, CellArraysIndexing, StaticArrays
    using JustPIC

    @init_parallel_stencil(AMDGPU, Float64, 3)

    import JustPIC:
        Euler, RungeKutta2, AbstractAdvectionIntegrator, Particles, PassiveMarkers
    import JustPIC: AbstractBackend, AMDGPUBackend

    macro myatomic(expr)
        return esc(
            quote
                AMDGPU.@atomic :monotonic $expr
            end,
        )
    end

    function JustPIC._3D.CA(::Type{AMDGPUBackend}, dims; eltype=Float64)
        return ROCCellArray{eltype}(undef, dims)
    end

    include(joinpath(@__DIR__, "../src/common.jl"))
    include(joinpath(@__DIR__, "../src/AMDGPUExt/CellArrays.jl"))

    JP_module = :(JustPIC._3D)
    ext_backend = :JustPIC.AMDGPUBackend
    TArray = :(AMDGPUBackend.ROCArray)
    include("common.jl")

end

end # module

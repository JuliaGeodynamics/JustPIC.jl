module JustPICAMDGPUExt

using AMDGPU
using JustPIC, CellArrays, StaticArrays

import JustPIC: AbstractBackend, AMDGPUBackend

JustPIC.TA(::Type{AMDGPUBackend}) = ROCArray

ROCCellArray(::Type{T}, ::UndefInitializer, dims::NTuple{N,Int}) where {T<:CellArrays.Cell,N} = CellArrays.CellArray{T,N,0,CUDA.ROCCellArrayArray{eltype(T),3}}(undef, dims)
ROCCellArray(::Type{T}, ::UndefInitializer, dims::Int...) where {T<:CellArrays.Cell} = ROCCellArray(T, undef, dims)

function AMDGPU.ROCArray(::Type{T}, particles::JustPIC.Particles) where {T<:Number}
    (; coords, index, nxcell, max_xcell, min_xcell, np) = particles
    coords_gpu = ntuple(i->ROCArray(T, coords[i]), Val(length(coords))) 
    return Particles(CUDABackend, coords_gpu, ROCArray(Bool, index), nxcell, max_xcell, min_xcell, np)
end

function AMDGPU.ROCArray(::Type{T}, phase_ratios::JustPIC.PhaseRatios) where {T<:Number}
    (; vertex, center) = phase_ratios
    return JustPIC.PhaseRatios(CUDABackend, ROCArray(T, center), ROCArray(T, vertex))
end

function AMDGPU.ROCArray(particles::JustPIC.Particles)
    (; coords, index, nxcell, max_xcell, min_xcell, np) = particles
    coords_gpu = ntuple(i->ROCArray(coords[i]), Val(length(coords))) 
    return Particles(CUDABackend, coords_gpu, ROCArray(index), nxcell, max_xcell, min_xcell, np)
end

function AMDGPU.ROCArray(phase_ratios::JustPIC.PhaseRatios)
    (; vertex, center) = phase_ratios
    return JustPIC.PhaseRatios(CUDABackend, ROCArray(center), ROCArray(vertex))
end

function AMDGPU.ROCArray(CA::CellArray) 
    ni     = size(CA)
    # Array initializations
    T_SArray = eltype(CA)
    CA_ROC = ROCCellArray(SVector{length(T_SArray), T}, undef, ni...)
    # copy data to the ROC CellArray
    tmp = if size(CA.data) != size(CA_ROC.data)
        ROCArray(permutedims(CA.data, (3, 2, 1)))
    else
        ROCArray(CA.data)
    end
    copyto!(CA_ROC.data, tmp)
    return CA_ROC
end

AMDGPU.ROCArray(particles::JustPIC.Particles{JustPIC.AMDGPUBackend})      = particles
AMDGPU.ROCArray(phase_ratios::JustPIC.PhaseRatios{JustPIC.AMDGPUBackend}) = phase_ratios
AMDGPU.ROCArray(CA::CellArray)                                            = AMDGPU.ROCArray(eltype(eltype(CA)), CA)

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


    function JustPIC._2D.Particles(
        coords,
        index::CellArray{StaticArraysCore.SVector{N1,Bool},2,0,ROCArray{Bool,N2}},
        nxcell,
        max_xcell,
        min_xcell,
        np,
    ) where {N1,N2}
        return Particles(AMDGPUBackend, coords, index, nxcell, max_xcell, min_xcell, np)
    end

    function JustPIC._2D.SubgridDiffusionCellArrays(particles::Particles{AMDGPUBackend})
        return SubgridDiffusionCellArrays(particles)
    end

    function JustPIC._2D.init_particles(
        ::Type{AMDGPUBackend}, nxcell, max_xcell, min_xcell, x, y
    )
        return init_particles(AMDGPUBackend, nxcell, max_xcell, min_xcell, x, y)
    end

    function JustPIC._2D.init_particles(
        ::Type{AMDGPUBackend},
        nxcell,
        max_xcell,
        min_xcell,
        coords::NTuple{2,AbstractArray},
        dxᵢ::NTuple{2,T},
        nᵢ::NTuple{2,I},
    ) where {T,I}
        return init_particles(AMDGPUBackend, nxcell, max_xcell, min_xcell, coords, dxᵢ, nᵢ)
    end

    function JustPIC._2D.advection!(
        particles::Particles{AMDGPUBackend},
        method::AbstractAdvectionIntegrator,
        V,
        grid_vxi::NTuple{N,NTuple{N,T}},
        dt,
    ) where {N,T}
        return advection!(particles, method, V, grid_vxi, dt)
    end

    function JustPIC._2D.advection_LinP!(
        particles::Particles{AMDGPUBackend},
        method::AbstractAdvectionIntegrator,
        V,
        grid_vxi::NTuple{N,NTuple{N,T}},
        dt,
    ) where {N,T}
        return advection_LinP!(particles, method, V, grid_vxi, dt)
    end

    function JustPIC._2D.advection_MQS!(
        particles::Particles{AMDGPUBackend},
        method::AbstractAdvectionIntegrator,
        V,
        grid_vxi::NTuple{N,NTuple{N,T}},
        dt,
    ) where {N,T}
        return advection_MQS!(particles, method, V, grid_vxi, dt)
    end

    function JustPIC._2D.centroid2particle!(
        Fp, xci, F::ROCArray, particles::Particles{AMDGPUBackend}
    )
        return centroid2particle!(Fp, xci, F, particles)
    end

    function JustPIC._2D.grid2particle!(
        Fp, xvi, F::ROCArray, particles::Particles{AMDGPUBackend}
    )
        return grid2particle!(Fp, xvi, F, particles)
    end

    function JustPIC._2D.particle2grid_centroid!(
        F::ROCArray, Fp, xi::NTuple, particles::Particles{AMDGPUBackend}
    )
        return particle2grid_centroid!(F, Fp, xi, particles)
    end

    function JustPIC._2D.particle2grid!(
        F::ROCArray, Fp, xi, particles::Particles{AMDGPUBackend}
    )
        return particle2grid!(F, Fp, xi, particles)
    end

    function JustPIC._2D.grid2particle_flip!(
        Fp, xvi, F::ROCArray, F0, particles::Particles{AMDGPUBackend}; α=0.0
    )
        return grid2particle_flip!(Fp, xvi, F, F0, particles; α=α)
    end

    function JustPIC._2D.inject_particles!(particles::Particles{AMDGPUBackend}, args, grid)
        return inject_particles!(particles, args, grid)
    end

    function JustPIC._2D.inject_particles_phase!(
        particles::Particles{AMDGPUBackend}, particles_phases, args, fields, grid
    )
        inject_particles_phase!(particles::Particles, particles_phases, args, fields, grid)
        return nothing
    end

    function JustPIC._2D.move_particles!(
        particles::Particles{AMDGPUBackend}, grid::NTuple{N}, args
    ) where {N}
        return move_particles!(particles, grid, args)
    end

    function JustPIC._2D.init_cell_arrays(
        particles::Particles{AMDGPUBackend}, V::Val{N}
    ) where {N}
        return init_cell_arrays(particles, V)
    end

    function JustPIC._2D.subgrid_diffusion!(
        pT,
        T_grid,
        ΔT_grid,
        subgrid_arrays,
        particles::Particles{AMDGPUBackend},
        xvi,
        di,
        dt;
        d=1.0,
    )
        subgrid_diffusion!(pT, T_grid, ΔT_grid, subgrid_arrays, particles, xvi, di, dt; d=d)
        return nothing
    end

    ## MakerChain

    function JustPIC._2D.advect_markerchain!(
        chain::MarkerChain{AMDGPUBackend},
        method::AbstractAdvectionIntegrator,
        V,
        grid_vxi,
        dt,
    )
        return advect_markerchain!(chain, method, V, grid_vxi, dt)
    end

    ## PassiveMarkers

    function JustPIC._2D.init_passive_markers(
        ::Type{AMDGPUBackend}, coords::NTuple{N,ROCArray}
    ) where {N}
        return init_passive_markers(AMDGPUBackend, coords)
    end

    function JustPIC._2D.advection!(
        particles::PassiveMarkers{AMDGPUBackend},
        method::AbstractAdvectionIntegrator,
        V::NTuple{N,ROCArray},
        grid_vxi,
        dt,
    ) where {N}
        return advection!(particles, method, V, grid_vxi, dt)
    end

    function JustPIC._2D.grid2particle!(
        Fp, xvi, F, particles::PassiveMarkers{AMDGPUBackend}
    )
        grid2particle!(Fp, xvi, F, particles)
        return nothing
    end

    function JustPIC._2D.grid2particle!(
        Fp::NTuple{N,ROCArray},
        xvi,
        F::NTuple{N,ROCArray},
        particles::PassiveMarkers{AMDGPUBackend},
    ) where {N}
        grid2particle!(Fp, xvi, F, particles)
        return nothing
    end

    function JustPIC._2D.particle2grid!(
        F, Fp, buffer, xi, particles::PassiveMarkers{AMDGPUBackend}
    )
        particle2grid!(F, Fp, buffer, xi, particles)
        return nothing
    end

    # Phase ratio kernels

    function JustPIC._2D.update_phase_ratios!(phase_ratios::JustPIC.PhaseRatios{AMDGPUBackend}, particles, xci, xvi, phases)
        phase_ratios_center!(phase_ratios, particles, xci, phases)
        phase_ratios_vertex!(phase_ratios, particles, xvi, phases)
        return nothing
    end

    function JustPIC._2D.PhaseRatios(
        ::Type{AMDGPUBackend}, nphases::Integer, ni::NTuple{N,Integer}
    ) where {N}
        return JustPIC._2D.PhaseRatios(Float64, AMDGPUBackend, nphases, ni)
    end

    function JustPIC._2D.PhaseRatios(
        ::Type{T}, ::Type{AMDGPUBackend}, nphases::Integer, ni::NTuple{N,Integer}
    ) where {N,T}
        center = cell_array(0.0, (nphases,), ni)
        vertex = cell_array(0.0, (nphases,), ni .+ 1)

        return JustPIC.PhaseRatios(AMDGPUBackend, center, vertex)
    end

    function JustPIC._2D.phase_ratios_center!(
        phase_ratios::JustPIC.PhaseRatios{AMDGPUBackend}, particles, xci, phases
    )
        ni = size(phases)
        di = compute_dx(xci)

        @parallel (@idx ni) phase_ratios_center_kernel!(
            phase_ratios.center, particles.coords, xci, di, phases
        )
        return nothing
    end

    function JustPIC._2D.phase_ratios_vertex!(
        phase_ratios::JustPIC.PhaseRatios{AMDGPUBackend}, particles, xvi, phases
    )
        ni = size(phases) .+ 1
        di = compute_dx(xvi)

        @parallel (@idx ni) phase_ratios_vertex_kernel!(
            phase_ratios.vertex, particles.coords, xvi, di, phases
        )
        return nothing
    end
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

    
    function JustPIC._3D.Particles(
        coords,
        index::CellArray{StaticArraysCore.SVector{N1,Bool},3,0,ROCArray{Bool,N2}},
        nxcell,
        max_xcell,
        min_xcell,
        np,
    ) where {N1,N2}
        return Particles(AMDGPUBackend, coords, index, nxcell, max_xcell, min_xcell, np)
    end

    function JustPIC._3D.SubgridDiffusionCellArrays(particles::Particles{AMDGPUBackend})
        return SubgridDiffusionCellArrays(particles)
    end

    function JustPIC._3D.init_particles(
        ::Type{AMDGPUBackend}, nxcell, max_xcell, min_xcell, x, y, z
    )
        return init_particles(AMDGPUBackend, nxcell, max_xcell, min_xcell, x, y, z)
    end

    function JustPIC._3D.init_particles(
        ::Type{AMDGPUBackend},
        nxcell,
        max_xcell,
        min_xcell,
        coords::NTuple{3,AbstractArray},
        dxᵢ::NTuple{3,T},
        nᵢ::NTuple{3,I},
    ) where {T,I}
        return init_particles(AMDGPUBackend, nxcell, max_xcell, min_xcell, coords, dxᵢ, nᵢ)
    end

    function JustPIC._3D.advection!(
        particles::Particles{AMDGPUBackend},
        method::AbstractAdvectionIntegrator,
        V,
        grid_vxi::NTuple{N,NTuple{N,T}},
        dt,
    ) where {N,T}
        return advection!(particles, method, V, grid_vxi, dt)
    end

    function JustPIC._3D.advection_LinP!(
        particles::Particles{AMDGPUBackend},
        method::AbstractAdvectionIntegrator,
        V,
        grid_vxi::NTuple{N,NTuple{N,T}},
        dt,
    ) where {N,T}
        return advection_LinP!(particles, method, V, grid_vxi, dt)
    end

    function JustPIC._3D.advection_MQS!(
        particles::Particles{AMDGPUBackend},
        method::AbstractAdvectionIntegrator,
        V,
        grid_vxi::NTuple{N,NTuple{N,T}},
        dt,
    ) where {N,T}
        return advection_MQS!(particles, method, V, grid_vxi, dt)
    end

    function JustPIC._3D.centroid2particle!(
        Fp, xci, F::ROCArray, particles::Particles{AMDGPUBackend}
    )
        return centroid2particle!(Fp, xci, F, particles)
    end

    function JustPIC._3D.grid2particle!(
        Fp, xvi, F::ROCArray, particles::Particles{AMDGPUBackend}
    )
        return grid2particle!(Fp, xvi, F, particles)
    end

    function JustPIC._3D.particle2grid_centroid!(
        F::ROCArray, Fp, xi::NTuple, particles::Particles{AMDGPUBackend}
    )
        return particle2grid_centroid!(F, Fp, xi, particles)
    end

    function JustPIC._3D.particle2grid!(
        F::ROCArray, Fp, xi, particles::Particles{AMDGPUBackend}
    )
        return particle2grid!(F, Fp, xi, particles)
    end

    function JustPIC._3D.grid2particle_flip!(Fp, xvi, F::ROCArray, F0, particles; α=0.0)
        return grid2particle_flip!(Fp, xvi, F, F0, particles; α=α)
    end

    function JustPIC._3D.inject_particles!(particles::Particles{AMDGPUBackend}, args, grid)
        return inject_particles!(particles, args, grid)
    end

    function JustPIC._3D.inject_particles_phase!(
        particles::Particles{AMDGPUBackend}, particles_phases, args, fields, grid
    )
        inject_particles_phase!(particles::Particles, particles_phases, args, fields, grid)
        return nothing
    end

    function JustPIC._3D.move_particles!(
        particles::Particles{AMDGPUBackend}, grid::NTuple{N}, args
    ) where {N}
        return move_particles!(particles, grid, args)
    end

    function JustPIC._3D.init_cell_arrays(
        particles::Particles{AMDGPUBackend}, V::Val{N}
    ) where {N}
        return init_cell_arrays(particles, V)
    end

    function JustPIC._3D.subgrid_diffusion!(
        pT,
        T_grid,
        ΔT_grid,
        subgrid_arrays,
        particles::Particles{AMDGPUBackend},
        xvi,
        di,
        dt;
        d=1.0,
    )
        subgrid_diffusion!(pT, T_grid, ΔT_grid, subgrid_arrays, particles, xvi, di, dt; d=d)
        return nothing
    end

    ## PassiveMarkers

    function JustPIC._3D.init_passive_markers(
        ::Type{AMDGPUBackend}, coords::NTuple{N,ROCArray}
    ) where {N}
        return init_passive_markers(AMDGPUBackend, coords)
    end

    function JustPIC._3D.advection!(
        particles::PassiveMarkers{AMDGPUBackend},
        method::AbstractAdvectionIntegrator,
        V::NTuple{N,ROCArray},
        grid_vxi,
        dt,
    ) where {N}
        return advection!(particles, method, V, grid_vxi, dt)
    end

    function JustPIC._3D.grid2particle!(Fp, xvi, F, particles::Particles{AMDGPUBackend})
        grid2particle!(Fp, xvi, F, particles)
        return nothing
    end

    function JustPIC._3D.grid2particle!(
        Fp::NTuple{N,ROCArray},
        xvi,
        F::NTuple{N,ROCArray},
        particles::Particles{AMDGPUBackend},
    ) where {N}
        grid2particle!(Fp, xvi, F, particles)
        return nothing
    end

    # Phase ratio kernels

    function JustPIC._3D.update_phase_ratios!(phase_ratios::JustPIC.PhaseRatios{AMDGPUBackend}, particles, xci, xvi, phases)
        phase_ratios_center!(phase_ratios, particles, xci, phases)
        phase_ratios_vertex!(phase_ratios, particles, xvi, phases)
        return nothing
    end

    function JustPIC._3D.PhaseRatios(
        ::Type{AMDGPUBackend}, nphases::Integer, ni::NTuple{N,Integer}
    ) where {N}
        return JustPIC._3D.PhaseRatios(Float64, AMDGPUBackend, nphases, ni)
    end

    function JustPIC._3D.PhaseRatios(
        ::Type{T}, ::Type{AMDGPUBackend}, nphases::Integer, ni::NTuple{N,Integer}
    ) where {N,T}
        center = cell_array(0.0, (nphases,), ni)
        vertex = cell_array(0.0, (nphases,), ni .+ 1)

        return JustPIC.PhaseRatios(AMDGPUBackend, center, vertex)
    end

    function JustPIC._3D.phase_ratios_center!(
        phase_ratios::JustPIC.PhaseRatios{AMDGPUBackend}, particles, xci, phases
    )
        ni = size(phases)
        di = compute_dx(xci)

        @parallel (@idx ni) phase_ratios_center_kernel!(
            phase_ratios.center, particles.coords, xci, di, phases
        )
        return nothing
    end

    function JustPIC._3D.phase_ratios_vertex!(
        phase_ratios::JustPIC.PhaseRatios{AMDGPUBackend}, particles, xvi, phases
    )
        ni = size(phases) .+ 1
        di = compute_dx(xvi)

        @parallel (@idx ni) phase_ratios_vertex_kernel!(
            phase_ratios.vertex, particles.coords, xvi, di, phases
        )
        return nothing
    end
end

end # module

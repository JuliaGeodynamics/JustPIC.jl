module JustPICAMDGPUExt

using AMDGPU
using JustPIC, CellArrays, StaticArrays

import JustPIC: AbstractBackend, AMDGPUBackend

JustPIC.TA(::Type{AMDGPUBackend}) = ROCArray

function ROCCellArray(
        ::Type{T}, ::UndefInitializer, dims::NTuple{N, Int}
    ) where {T <: CellArrays.Cell, N}
    return CellArrays.CellArray{T, N, 0, AMDGPU.ROCCellArrayArray{eltype(T), 3}}(undef, dims)
end
function ROCCellArray(
        ::Type{T}, ::UndefInitializer, dims::Int...
    ) where {T <: CellArrays.Cell}
    return ROCCellArray(T, undef, dims)
end

function AMDGPU.ROCArray(::Type{T}, particles::JustPIC.Particles) where {T <: Number}
    (; coords, index, nxcell, max_xcell, min_xcell, np) = particles
    coords_gpu = ntuple(i -> ROCArray(T, coords[i]), Val(length(coords)))
    return Particles(
        AMDGPUBackend, coords_gpu, ROCArray(Bool, index), nxcell, max_xcell, min_xcell, np
    )
end

function AMDGPU.ROCArray(::Type{T}, chain::JustPIC.MarkerChain) where {T <: Number}
    (;
        cell_vertices, coords, coords0, h_vertices, h_vertices0, index, max_xcell, min_xcell,
    ) = chain
    coords_gpu = ntuple(i -> ROCArray(T, coords[i]), Val(length(coords)))
    coords0_gpu = ntuple(i -> ROCArray(T, coords0[i]), Val(length(coords0)))
    return MarkerChain(
        CUDABackend,
        coords_gpu,
        coords0_gpu,
        ROCArray(h_vertices),
        ROCArray(h_vertices0),
        cell_vertices,
        ROCArray(Bool, index),
        max_xcell,
        min_xcell,
    )
end

function AMDGPU.ROCArray(::Type{T}, phase_ratios::JustPIC.PhaseRatios) where {T <: Number}
    (; center, vertex, Vx, Vy, Vz, yz, xz, xy) = phase_ratios
    return JustPIC.PhaseRatios(
        AMDGPUBackend,
        ROCArray(T, center),
        ROCArray(T, vertex),
        ROCArray(T, Vx),
        ROCArray(T, Vy),
        ROCArray(T, Vz),
        ROCArray(T, yz),
        ROCArray(T, xz),
        ROCArray(T, xy),
    )
end

function AMDGPU.ROCArray(phase_ratios::JustPIC.PhaseRatios)
    (; center, vertex, Vx, Vy, Vz, yz, xz, xy) = phase_ratios
    return JustPIC.PhaseRatios(
        AMDGPUBackend,
        ROCArray(center),
        ROCArray(vertex),
        ROCArray(Vx),
        ROCArray(Vy),
        ROCArray(Vz),
        ROCArray(yz),
        ROCArray(xz),
        ROCArray(xy),
    )
end

function AMDGPU.ROCArray(particles::JustPIC.Particles)
    (; coords, index, nxcell, max_xcell, min_xcell, np) = particles
    coords_gpu = ntuple(i -> ROCArray(coords[i]), Val(length(coords)))
    return Particles(
        AMDGPUBackend, coords_gpu, ROCArray(index), nxcell, max_xcell, min_xcell, np
    )
end

function AMDGPU.ROCArray(chain::JustPIC.MarkerChain)
    (;
        cell_vertices, coords, coords0, h_vertices, h_vertices0, index, max_xcell, min_xcell,
    ) = chain
    coords_gpu = ntuple(i -> ROCArray(coords[i]), Val(length(coords)))
    coords0_gpu = ntuple(i -> ROCArray(coords0[i]), Val(length(coords0)))
    return MarkerChain(
        CUDABackend,
        coords_gpu,
        coords0_gpu,
        ROCArray(h_vertices),
        ROCArray(h_vertices0),
        cell_vertices,
        ROCArray(Bool, index),
        max_xcell,
        min_xcell,
    )
end

function AMDGPU.ROCArray(phase_ratios::JustPIC.PhaseRatios)
    (; vertex, center) = phase_ratios
    return JustPIC.PhaseRatios(AMDGPUBackend, ROCArray(center), ROCArray(vertex))
end

function AMDGPU.ROCArray(::Type{T}, CA::CellArray) where {T <: Number}
    ni = size(CA)
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

AMDGPU.ROCArray(particles::JustPIC.Particles{JustPIC.AMDGPUBackend}) = particles
AMDGPU.ROCArray(phase_ratios::JustPIC.PhaseRatios{JustPIC.AMDGPUBackend}) = phase_ratios
AMDGPU.ROCArray(CA::CellArray) = AMDGPU.ROCArray(eltype(eltype(CA)), CA)

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

    function JustPIC._2D.CA(::Type{AMDGPUBackend}, dims; eltype = Float64)
        return ROCCellArray{eltype}(undef, dims)
    end

    include(joinpath(@__DIR__, "../src/common.jl"))
    include(joinpath(@__DIR__, "../src/AMDGPUExt/CellArrays.jl"))

    # halo update
    function JustPIC._2D.update_cell_halo!(
            x::Vararg{CellArray{S, N, D, ROCArray{T, nD}}, NA}
        ) where {NA, S, N, D, T, nD}
        return update_cell_halo!(x...)
    end
    function JustPIC._2D.update_cell_halo!(
            x::Vararg{CellArray{S, N, D, ROCArray{T, nD, B}}, NA}
        ) where {NA, S, N, D, T, nD, B}
        return update_cell_halo!(x...)
    end

    # Conversions
    function JustPIC._2D.Particles(
            coords,
            index::CellArray{StaticArraysCore.SVector{N1, Bool}, 2, 0, ROCArray{Bool, N2}},
            nxcell,
            max_xcell,
            min_xcell,
            np,
        ) where {N1, N2}
        return Particles(AMDGPUBackend, coords, index, nxcell, max_xcell, min_xcell, np)
    end

    function JustPIC._2D.Particles(
            coords,
            index::CellArray{StaticArraysCore.SVector{N1, Bool}, 2, 0, ROCArray{Bool, N2, B}},
            nxcell,
            max_xcell,
            min_xcell,
            np,
        ) where {B, N1, N2}
        return Particles(AMDGPUBackend, coords, index, nxcell, max_xcell, min_xcell, np)
    end

    function JustPIC._2D.SubgridDiffusionCellArrays(
            particles::Particles{AMDGPUBackend}; loc::Symbol = :vertex
        )
        return SubgridDiffusionCellArrays(particles; loc = loc)
    end

    function JustPIC._2D.init_particles(
            ::Type{AMDGPUBackend}, nxcell, max_xcell, min_xcell, x, y; buffer = 1 - 1.0e-5
        )
        return init_particles(
            AMDGPUBackend, nxcell, max_xcell, min_xcell, x, y; buffer = buffer
        )
    end

    function JustPIC._2D.init_particles(
            ::Type{AMDGPUBackend},
            nxcell,
            max_xcell,
            min_xcell,
            coords::NTuple{3, AbstractArray},
            dxᵢ::NTuple{3, T},
            nᵢ::NTuple{3, I};
            buffer = 1 - 1.0e-5,
        ) where {T, I}
        return init_particles(
            AMDGPUBackend, nxcell, max_xcell, min_xcell, coords, dxᵢ, nᵢ; buffer = buffer
        )
    end

    function JustPIC._2D.advection!(
            particles::Particles{AMDGPUBackend},
            method::AbstractAdvectionIntegrator,
            V,
            grid_vxi::NTuple{N, NTuple{N, T}},
            dt,
        ) where {N, T}
        return advection!(particles, method, V, grid_vxi, dt)
    end

    function JustPIC._2D.advection_LinP!(
            particles::Particles{AMDGPUBackend},
            method::AbstractAdvectionIntegrator,
            V,
            grid_vxi::NTuple{N, NTuple{N, T}},
            dt,
        ) where {N, T}
        return advection_LinP!(particles, method, V, grid_vxi, dt)
    end

    function JustPIC._2D.advection_MQS!(
            particles::Particles{AMDGPUBackend},
            method::AbstractAdvectionIntegrator,
            V,
            grid_vxi::NTuple{N, NTuple{N, T}},
            dt,
        ) where {N, T}
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

    function JustPIC._2D.particle2centroid!(
            F::ROCArray, Fp, xi::NTuple, particles::Particles{AMDGPUBackend}
        )
        return particle2centroid!(F, Fp, xi, particles)
    end

    function JustPIC._2D.particle2grid!(
            F::ROCArray, Fp, xi, particles::Particles{AMDGPUBackend}
        )
        return particle2grid!(F, Fp, xi, particles)
    end

    function JustPIC._2D.grid2particle_flip!(
            Fp, xvi, F::ROCArray, F0, particles::Particles{AMDGPUBackend}; α = 0.0
        )
        return grid2particle_flip!(Fp, xvi, F, F0, particles; α = α)
    end

    function JustPIC._2D.inject_particles!(
            particles::Particles{AMDGPUBackend}, args, grid::NTuple{N}
        ) where {N}
        return inject_particles!(particles, args, grid)
    end

    function JustPIC._2D.force_injection!(particles::Particles{AMDGPUBackend}, p_new, fields::NTuple{N, Any}, values::NTuple{N, Any}) where {N}
        force_injection!(particles, p_new, fields, values)
        return nothing
    end

    JustPIC._2D.force_injection!(particles::Particles{AMDGPUBackend}, p_new) = force_injection!(particles, p_new, (), ())

    function JustPIC._2D.inject_particles_phase!(
            particles::Particles{AMDGPUBackend}, particles_phases, args, fields, grid::NTuple{N}
        ) where {N}
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
            d = 1.0,
        )
        subgrid_diffusion!(pT, T_grid, ΔT_grid, subgrid_arrays, particles, xvi, di, dt; d = d)
        return nothing
    end

    function JustPIC._2D.subgrid_diffusion_centroid!(
            pT,
            T_grid,
            ΔT_grid,
            subgrid_arrays,
            particles::Particles{AMDGPUBackend},
            xvi,
            di,
            dt;
            d = 1.0,
        )
        subgrid_diffusion_centroid!(
            pT, T_grid, ΔT_grid, subgrid_arrays, particles, xvi, di, dt; d = d
        )
        return nothing
    end

    ## MakerChain

    function JustPIC._2D.init_markerchain(
            ::Type{AMDGPUBackend}, nxcell, min_xcell, max_xcell, xv, initial_elevation
        )
        return init_markerchain(
            AMDGPUBackend, nxcell, min_xcell, max_xcell, xv, initial_elevation
        )
    end

    function JustPIC._2D.fill_chain_from_chain!(
            chain::MarkerChain{AMDGPUBackend}, topo_x, topo_y
        )
        return fill_chain_from_chain!(chain, topo_x, topo_y)
    end

    function JustPIC._2D.compute_topography_vertex!(chain::MarkerChain{AMDGPUBackend})
        compute_topography_vertex!(chain)
        return nothing
    end

    function JustPIC._2D.reconstruct_chain_from_vertices!(chain::MarkerChain{AMDGPUBackend})
        reconstruct_chain_from_vertices!(chain)
        return nothing
    end

    function JustPIC._2D.fill_chain_from_vertices!(
            chain::MarkerChain{AMDGPUBackend}, topo_y
        )
        fill_chain_from_vertices!(chain::MarkerChain, topo_y)
        return nothing
    end

    function JustPIC._2D.advect_markerchain!(
            chain::MarkerChain{AMDGPUBackend},
            method::AbstractAdvectionIntegrator,
            V,
            grid_vxi,
            dt,
        )
        return advect_markerchain!(chain, method, V, grid_vxi, dt)
    end

    function JustPIC._2D.compute_rock_fraction!(
            ratios, chain::MarkerChain{AMDGPUBackend}, xvi, dxi
        )
        compute_rock_fraction!(ratios, chain, xvi, dxi)
        return nothing
    end

    function JustPIC._2D.interpolate_velocity_to_markerchain!(chain::MarkerChain{AMDGPUBackend}, chain_V, V, grid_vi::NTuple{N, NTuple{N, T}}) where {N, T}
        interpolate_velocity_to_markerchain!(chain, chain_V, V, grid_vi)
        return nothing
    end

    ## PassiveMarkers

    function JustPIC._2D.init_passive_markers(
            ::Type{AMDGPUBackend}, coords::NTuple{N, ROCArray}
        ) where {N}
        return init_passive_markers(AMDGPUBackend, coords)
    end

    function JustPIC._2D.advection!(
            particles::PassiveMarkers{AMDGPUBackend},
            method::AbstractAdvectionIntegrator,
            V::NTuple{N, ROCArray},
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
            Fp::NTuple{N, ROCArray},
            xvi,
            F::NTuple{N, ROCArray},
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

    function JustPIC._2D.update_phase_ratios!(
            phase_ratios::JustPIC.PhaseRatios{AMDGPUBackend, T}, particles, xci, xvi, phases
        ) where {T <: AbstractMatrix}
        phase_ratios_center!(phase_ratios, particles, xci, phases)
        phase_ratios_vertex!(phase_ratios, particles, xvi, phases)
        # velocity nodes
        phase_ratios_face!(phase_ratios.Vx, particles, xci, phases, :x)
        phase_ratios_face!(phase_ratios.Vy, particles, xci, phases, :y)
        return nothing
    end

    function JustPIC._2D.PhaseRatios(
            ::Type{AMDGPUBackend}, nphases::Integer, ni::NTuple{N, Integer}
        ) where {N}
        return JustPIC._2D.PhaseRatios(Float64, AMDGPUBackend, nphases, ni)
    end

    function JustPIC._2D.PhaseRatios(
            ::Type{T}, ::Type{AMDGPUBackend}, nphases::Integer, ni::NTuple{2, Integer}
        ) where {T}
        nx, ny = ni

        center = cell_array(zero(T), (nphases,), ni)
        vertex = cell_array(zero(T), (nphases,), ni .+ 1)
        Vx = cell_array(zero(T), (nphases,), (nx + 1, ny))
        Vy = cell_array(zero(T), (nphases,), (nx, ny + 1))
        dummy = cell_array(zero(T), (nphases,), (1, 1)) # because it cant be a Union{T, Nothing} type on the GPU....

        return JustPIC.PhaseRatios(
            AMDGPUBackend, center, vertex, Vx, Vy, dummy, dummy, dummy, dummy
        )
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

    function JustPIC._2D.phase_ratios_midpoint!(
            phase_midpoint,
            particles::Particles{AMDGPUBackend},
            xci::NTuple{N},
            phases,
            dimension,
        ) where {N}
        phase_ratios_midpoint!(phase_midpoint, particles, xci, phases, dimension)
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

    function JustPIC._3D.CA(::Type{AMDGPUBackend}, dims; eltype = Float64)
        return ROCCellArray{eltype}(undef, dims)
    end

    include(joinpath(@__DIR__, "../src/common.jl"))
    include(joinpath(@__DIR__, "../src/AMDGPUExt/CellArrays.jl"))

    # halo update
    function JustPIC._3D.update_cell_halo!(
            x::Vararg{CellArray{S, N, D, ROCArray{T, nD}}, NA}
        ) where {NA, S, N, D, T, nD}
        return update_cell_halo!(x...)
    end
    function JustPIC._3D.update_cell_halo!(
            x::Vararg{CellArray{S, N, D, ROCArray{T, nD, B}}, NA}
        ) where {NA, S, N, D, T, nD, B}
        return update_cell_halo!(x...)
    end

    function JustPIC._3D.Particles(
            coords,
            index::CellArray{StaticArraysCore.SVector{N1, Bool}, 3, 0, ROCArray{Bool, N2}},
            nxcell,
            max_xcell,
            min_xcell,
            np,
        ) where {N1, N2}
        return Particles(AMDGPUBackend, coords, index, nxcell, max_xcell, min_xcell, np)
    end

    function JustPIC._3D.Particles(
            coords,
            index::CellArray{StaticArraysCore.SVector{N1, Bool}, 3, 0, ROCArray{Bool, N2, B}},
            nxcell,
            max_xcell,
            min_xcell,
            np,
        ) where {B, N1, N2}
        return Particles(AMDGPUBackend, coords, index, nxcell, max_xcell, min_xcell, np)
    end

    function JustPIC._3D.SubgridDiffusionCellArrays(
            particles::Particles{AMDGPUBackend}; loc::Symbol = :vertex
        )
        return SubgridDiffusionCellArrays(particles; loc = loc)
    end

    function JustPIC._3D.init_particles(
            ::Type{AMDGPUBackend}, nxcell, max_xcell, min_xcell, x, y, z; buffer = 1 - 1.0e-5
        )
        return init_particles(
            AMDGPUBackend, nxcell, max_xcell, min_xcell, x, y, z; buffer = buffer
        )
    end

    function JustPIC._3D.init_particles(
            ::Type{AMDGPUBackend},
            nxcell,
            max_xcell,
            min_xcell,
            coords::NTuple{3, AbstractArray},
            dxᵢ::NTuple{3, T},
            nᵢ::NTuple{3, I};
            buffer = 1 - 1.0e-5,
        ) where {T, I}
        return init_particles(
            AMDGPUBackend, nxcell, max_xcell, min_xcell, coords, dxᵢ, nᵢ; buffer = buffer
        )
    end

    function JustPIC._3D.advection!(
            particles::Particles{AMDGPUBackend},
            method::AbstractAdvectionIntegrator,
            V,
            grid_vxi::NTuple{N, NTuple{N, T}},
            dt,
        ) where {N, T}
        return advection!(particles, method, V, grid_vxi, dt)
    end

    function JustPIC._3D.advection_LinP!(
            particles::Particles{AMDGPUBackend},
            method::AbstractAdvectionIntegrator,
            V,
            grid_vxi::NTuple{N, NTuple{N, T}},
            dt,
        ) where {N, T}
        return advection_LinP!(particles, method, V, grid_vxi, dt)
    end

    function JustPIC._3D.advection_MQS!(
            particles::Particles{AMDGPUBackend},
            method::AbstractAdvectionIntegrator,
            V,
            grid_vxi::NTuple{N, NTuple{N, T}},
            dt,
        ) where {N, T}
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

    function JustPIC._3D.particle2centroid!(
            F::ROCArray, Fp, xi::NTuple, particles::Particles{AMDGPUBackend}
        )
        return particle2centroid!(F, Fp, xi, particles)
    end

    function JustPIC._3D.particle2grid!(
            F::ROCArray, Fp, xi, particles::Particles{AMDGPUBackend}
        )
        return particle2grid!(F, Fp, xi, particles)
    end

    function JustPIC._3D.grid2particle_flip!(Fp, xvi, F::ROCArray, F0, particles; α = 0.0)
        return grid2particle_flip!(Fp, xvi, F, F0, particles; α = α)
    end

    function JustPIC._3D.inject_particles!(
            particles::Particles{AMDGPUBackend}, args, grid::NTuple{N}
        ) where {N}
        return inject_particles!(particles, args, grid)
    end

    function JustPIC._3D.force_injection!(particles::Particles{AMDGPUBackend}, p_new, fields::NTuple{N, Any}, values::NTuple{N, Any}) where {N}
        force_injection!(particles, p_new, fields, values)
        return nothing
    end

    JustPIC._3D.force_injection!(particles::Particles{AMDGPUBackend}, p_new) = force_injection!(particles, p_new, (), ())

    function JustPIC._3D.inject_particles_phase!(
            particles::Particles{AMDGPUBackend}, particles_phases, args, fields, grid::NTuple{N}
        ) where {N}
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
            d = 1.0,
        )
        subgrid_diffusion!(pT, T_grid, ΔT_grid, subgrid_arrays, particles, xvi, di, dt; d = d)
        return nothing
    end

    function JustPIC._3D.subgrid_diffusion_centroid!(
            pT,
            T_grid,
            ΔT_grid,
            subgrid_arrays,
            particles::Particles{AMDGPUBackend},
            xvi,
            di,
            dt;
            d = 1.0,
        )
        subgrid_diffusion_centroid!(
            pT, T_grid, ΔT_grid, subgrid_arrays, particles, xvi, di, dt; d = d
        )
        return nothing
    end

    ## PassiveMarkers

    function JustPIC._3D.init_passive_markers(
            ::Type{AMDGPUBackend}, coords::NTuple{N, ROCArray}
        ) where {N}
        return init_passive_markers(AMDGPUBackend, coords)
    end

    function JustPIC._3D.advection!(
            particles::PassiveMarkers{AMDGPUBackend},
            method::AbstractAdvectionIntegrator,
            V::NTuple{N, ROCArray},
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
            Fp::NTuple{N, ROCArray},
            xvi,
            F::NTuple{N, ROCArray},
            particles::Particles{AMDGPUBackend},
        ) where {N}
        grid2particle!(Fp, xvi, F, particles)
        return nothing
    end

    # Phase ratio kernels

    function JustPIC._3D.update_phase_ratios!(
            phase_ratios::JustPIC.PhaseRatios{AMDGPUBackend, T}, particles, xci, xvi, phases
        ) where {T <: AbstractArray}
        phase_ratios_center!(phase_ratios, particles, xci, phases)
        phase_ratios_vertex!(phase_ratios, particles, xvi, phases)
        # velocity nodes
        phase_ratios_face!(phase_ratios.Vx, particles, xci, phases, :x)
        phase_ratios_face!(phase_ratios.Vy, particles, xci, phases, :y)
        phase_ratios_face!(phase_ratios.Vz, particles, xci, phases, :z)
        # shear stress nodes
        phase_ratios_midpoint!(phase_ratios.xy, particles, xci, phases, :xy)
        phase_ratios_midpoint!(phase_ratios.yz, particles, xci, phases, :yz)
        phase_ratios_midpoint!(phase_ratios.xz, particles, xci, phases, :xz)
        return nothing
    end

    function JustPIC._3D.PhaseRatios(
            ::Type{AMDGPUBackend}, nphases::Integer, ni::NTuple{N, Integer}
        ) where {N}
        return JustPIC._3D.PhaseRatios(Float64, AMDGPUBackend, nphases, ni)
    end

    function JustPIC._3D.PhaseRatios(
            ::Type{T}, ::Type{AMDGPUBackend}, nphases::Integer, ni::NTuple{3, Integer}
        ) where {T}
        nx, ny, nz = ni

        center = cell_array(zero(T), (nphases,), ni)
        vertex = cell_array(zero(T), (nphases,), ni .+ 1)
        Vx = cell_array(zero(T), (nphases,), (nx + 1, ny, nz))
        Vy = cell_array(zero(T), (nphases,), (nx, ny + 1, nz))
        Vz = cell_array(zero(T), (nphases,), (nx, ny, nz + 1))
        yz = cell_array(zero(T), (nphases,), (nx, ny + 1, nz + 1))
        xz = cell_array(zero(T), (nphases,), (nx + 1, ny, nz + 1))
        xy = cell_array(zero(T), (nphases,), (nx + 1, ny + 1, nz))

        return JustPIC.PhaseRatios(AMDGPUBackend, center, vertex, Vx, Vy, Vz, yz, xz, xy)
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

    function JustPIC._3D.phase_ratios_midpoint!(
            phase_midpoint,
            particles::Particles{AMDGPUBackend},
            xci::NTuple{N},
            phases,
            dimension,
        ) where {N}
        phase_ratios_midpoint!(phase_midpoint, particles, xci, phases, dimension)
        return nothing
    end
end

end # module

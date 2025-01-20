module JustPICCUDAExt

using CUDA
using JustPIC, CellArrays, StaticArrays

JustPIC.TA(::Type{CUDABackend}) = CuArray

function CuCellArray(
    ::Type{T}, ::UndefInitializer, dims::NTuple{N,Int}
) where {T<:CellArrays.Cell,N}
    return CellArrays.CellArray{T,N,0,CUDA.CuArray{eltype(T),3}}(undef, dims)
end
function CuCellArray(::Type{T}, ::UndefInitializer, dims::Int...) where {T<:CellArrays.Cell}
    return CuCellArray(T, undef, dims)
end

function CUDA.CuArray(::Type{T}, particles::JustPIC.Particles) where {T<:Number}
    (; coords, index, nxcell, max_xcell, min_xcell, np) = particles
    coords_gpu = ntuple(i -> CuArray(T, coords[i]), Val(length(coords)))
    return Particles(
        CUDABackend, coords_gpu, CuArray(Bool, index), nxcell, max_xcell, min_xcell, np
    )
end

function CUDA.CuArray(::Type{T}, chain::JustPIC.MarkerChain) where {T<:Number}
    (;
        cell_vertices, coords, coords0, h_vertices, h_vertices0, index, max_xcell, min_xcell
    ) = chain
    coords_gpu = ntuple(i -> CuArray(T, coords[i]), Val(length(coords)))
    coords0_gpu = ntuple(i -> CuArray(T, coords0[i]), Val(length(coords0)))
    return MarkerChain(
        CUDABackend,
        coords_gpu,
        coords0_gpu,
        CuArray(h_vertices),
        CuArray(h_vertices0),
        cell_vertices,
        CuArray(Bool, index),
        max_xcell,
        min_xcell,
    )
end

function CUDA.CuArray(::Type{T}, phase_ratios::JustPIC.PhaseRatios) where {T<:Number}
    (; center, vertex, Vx, Vy, Vz, yz, xz, xy) = phase_ratios
    return JustPIC.PhaseRatios(
        CUDABackend,
        CuArray(T, center),
        CuArray(T, vertex),
        CuArray(T, Vx),
        CuArray(T, Vy),
        CuArray(T, Vz),
        CuArray(T, yz),
        CuArray(T, xz),
        CuArray(T, xy),
    )
end

function CUDA.CuArray(phase_ratios::JustPIC.PhaseRatios)
    (; center, vertex, Vx, Vy, Vz, yz, xz, xy) = phase_ratios
    return JustPIC.PhaseRatios(
        CUDABackend,
        CuArray(center),
        CuArray(vertex),
        CuArray(Vx),
        CuArray(Vy),
        CuArray(Vz),
        CuArray(yz),
        CuArray(xz),
        CuArray(xy),
    )
end

function CUDA.CuArray(particles::JustPIC.Particles)
    (; coords, index, nxcell, max_xcell, min_xcell, np) = particles
    coords_gpu = ntuple(i -> CuArray(coords[i]), Val(length(coords)))
    return Particles(
        CUDABackend, coords_gpu, CuArray(index), nxcell, max_xcell, min_xcell, np
    )
end

function CUDA.CuArray(chain::JustPIC.MarkerChain)
    (;
        cell_vertices, coords, coords0, h_vertices, h_vertices0, index, max_xcell, min_xcell
    ) = chain
    coords_gpu = ntuple(i -> CuArray(coords[i]), Val(length(coords)))
    coords0_gpu = ntuple(i -> CuArray(coords0[i]), Val(length(coords0)))
    return MarkerChain(
        CUDABackend,
        coords_gpu,
        coords0_gpu,
        CuArray(h_vertices),
        CuArray(h_vertices0),
        cell_vertices,
        CuArray(Bool, index),
        max_xcell,
        min_xcell,
    )
end

function CUDA.CuArray(::Type{T}, CA::CellArray) where {T<:Number}
    ni = size(CA)
    # Array initializations
    T_SArray = eltype(CA)
    CA_CUDA = CuCellArray(SVector{length(T_SArray),T}, undef, ni...)
    # copy data to the CUDA CellArray
    tmp = if size(CA.data) != size(CA_CUDA.data)
        CuArray(permutedims(CA.data, (3, 2, 1)))
    else
        CuArray(CA.data)
    end
    copyto!(CA_CUDA.data, tmp)
    return CA_CUDA
end

CUDA.CuArray(particles::JustPIC.Particles{CUDABackend}) = particles
CUDA.CuArray(phase_ratios::JustPIC.PhaseRatios{CUDABackend}) = phase_ratios
CUDA.CuArray(CA::CellArray) = CUDA.CuArray(eltype(eltype(CA)), CA)

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

    # halo update
    function JustPIC._2D.update_cell_halo!(
        x::Vararg{CellArray{S,N,D,CuArray{T,nD}},NA}
    ) where {NA,S,N,D,T,nD}
        return update_cell_halo!(x...)
    end
    function JustPIC._2D.update_cell_halo!(
        x::Vararg{CellArray{S,N,D,CuArray{T,nD,B}},NA}
    ) where {NA,S,N,D,T,nD,B}
        return update_cell_halo!(x...)
    end

    # Conversions
    function JustPIC._2D.Particles(
        coords,
        index::CellArray{StaticArraysCore.SVector{N1,Bool},2,0,CuArray{Bool,N2}},
        nxcell,
        max_xcell,
        min_xcell,
        np,
    ) where {N1,N2}
        return Particles(CUDABackend, coords, index, nxcell, max_xcell, min_xcell, np)
    end

    function JustPIC._2D.Particles(
        coords,
        index::CellArray{StaticArraysCore.SVector{N1,Bool},2,0,CuArray{Bool,N2,B}},
        nxcell,
        max_xcell,
        min_xcell,
        np,
    ) where {B,N1,N2}
        return Particles(CUDABackend, coords, index, nxcell, max_xcell, min_xcell, np)
    end

    function JustPIC._2D.SubgridDiffusionCellArrays(
        particles::Particles{CUDABackend}; loc::Symbol=:vertex
    )
        return SubgridDiffusionCellArrays(particles; loc=loc)
    end

    function JustPIC._2D.init_particles(
        ::Type{CUDABackend}, nxcell, max_xcell, min_xcell, x, y; buffer=1 - 1e-5
    )
        return init_particles(
            CUDABackend, nxcell, max_xcell, min_xcell, x, y; buffer=buffer
        )
    end

    function JustPIC._2D.init_particles(
        ::Type{CUDABackend},
        nxcell,
        max_xcell,
        min_xcell,
        coords::NTuple{2,AbstractArray},
        dxᵢ::NTuple{2,T},
        nᵢ::NTuple{2,I};
        buffer=1 - 1e-5,
    ) where {T,I}
        return init_particles(
            CUDABackend, nxcell, max_xcell, min_xcell, coords, dxᵢ, nᵢ; buffer=buffer
        )
    end

    function JustPIC._2D.advection!(
        particles::Particles{CUDABackend},
        method::AbstractAdvectionIntegrator,
        V,
        grid_vxi::NTuple{N,NTuple{N,T}},
        dt,
    ) where {N,T}
        return advection!(particles, method, V, grid_vxi, dt)
    end

    function JustPIC._2D.advection_LinP!(
        particles::Particles{CUDABackend},
        method::AbstractAdvectionIntegrator,
        V,
        grid_vxi::NTuple{N,NTuple{N,T}},
        dt,
    ) where {N,T}
        return advection_LinP!(particles, method, V, grid_vxi, dt)
    end

    function JustPIC._2D.advection_MQS!(
        particles::Particles{CUDABackend},
        method::AbstractAdvectionIntegrator,
        V,
        grid_vxi::NTuple{N,NTuple{N,T}},
        dt,
    ) where {N,T}
        return advection_MQS!(particles, method, V, grid_vxi, dt)
    end

    function JustPIC._2D.centroid2particle!(
        Fp, xci, F::CuArray, particles::Particles{CUDABackend}
    )
        return centroid2particle!(Fp, xci, F, particles)
    end

    function JustPIC._2D.grid2particle!(
        Fp, xvi, F::CuArray, particles::Particles{CUDABackend}
    )
        return grid2particle!(Fp, xvi, F, particles)
    end

    function JustPIC._2D.particle2centroid!(
        F::CuArray, Fp, xi::NTuple, particles::Particles{CUDABackend}
    )
        return particle2centroid!(F, Fp, xi, particles)
    end

    function JustPIC._2D.particle2grid!(F::CuArray, Fp, xi, particles)
        return particle2grid!(F, Fp, xi, particles)
    end

    function JustPIC._2D.grid2particle_flip!(Fp, xvi, F::CuArray, F0, particles; α=0.0)
        return grid2particle_flip!(Fp, xvi, F, F0, particles; α=α)
    end

    function JustPIC._2D.inject_particles!(
        particles::Particles{CUDABackend}, args, grid::NTuple{N}
    ) where {N}
        return inject_particles!(particles, args, grid)
    end

    function JustPIC._2D.inject_particles_phase!(
        particles::Particles{CUDABackend}, particles_phases, args, fields, grid::NTuple{N}
    ) where {N}
        inject_particles_phase!(particles::Particles, particles_phases, args, fields, grid)
        return nothing
    end

    function JustPIC._2D.move_particles!(
        particles::Particles{CUDABackend}, grid::NTuple{N}, args
    ) where {N}
        return move_particles!(particles, grid, args)
    end

    function JustPIC._2D.init_cell_arrays(
        particles::Particles{CUDABackend}, V::Val{N}
    ) where {N}
        return init_cell_arrays(particles, V)
    end

    function JustPIC._2D.subgrid_diffusion!(
        pT,
        T_grid,
        ΔT_grid,
        subgrid_arrays,
        particles::Particles{CUDABackend},
        xvi,
        di,
        dt;
        d=1.0,
    )
        subgrid_diffusion!(pT, T_grid, ΔT_grid, subgrid_arrays, particles, xvi, di, dt; d=d)
        return nothing
    end

    function JustPIC._2D.subgrid_diffusion_centroid!(
        pT,
        T_grid,
        ΔT_grid,
        subgrid_arrays,
        particles::Particles{CUDABackend},
        xvi,
        di,
        dt;
        d=1.0,
    )
        subgrid_diffusion_centroid!(
            pT, T_grid, ΔT_grid, subgrid_arrays, particles, xvi, di, dt; d=d
        )
        return nothing
    end

    ## MakerChain

    function JustPIC._2D.init_markerchain(
        ::Type{CUDABackend}, nxcell, min_xcell, max_xcell, xv, initial_elevation
    )
        return init_markerchain(
            CUDABackend, nxcell, min_xcell, max_xcell, xv, initial_elevation
        )
    end

    function JustPIC._2D.fill_chain_from_chain!(
        chain::MarkerChain{CUDABackend}, topo_x, topo_y
    )
        fill_chain_from_chain!(chain, topo_x, topo_y)
        return nothing
    end

    function JustPIC._2D.compute_topography_vertex!(chain::MarkerChain{CUDABackend})
        compute_topography_vertex!(chain)
        return nothing
    end

    function JustPIC._2D.reconstruct_chain_from_vertices!(chain::MarkerChain{CUDABackend})
        reconstruct_chain_from_vertices!(chain)
        return nothing
    end

    function JustPIC._2D.fill_chain_from_vertices!(chain::MarkerChain{CUDABackend}, topo_y)
        fill_chain_from_vertices!(chain::MarkerChain, topo_y)
        return nothing
    end

    function JustPIC._2D.advect_markerchain!(
        chain::MarkerChain{CUDABackend},
        method::AbstractAdvectionIntegrator,
        V,
        grid_vxi,
        dt,
    )
        return advect_markerchain!(chain, method, V, grid_vxi, dt)
    end

    function JustPIC._2D.compute_rock_fraction!(
        ratios, chain::MarkerChain{CUDABackend}, xvi, dxi
    )
        compute_rock_fraction!(ratios, chain, xvi, dxi)
        return nothing
    end

    ## PassiveMarkers

    function JustPIC._2D.init_passive_markers(
        ::Type{CUDABackend}, coords::NTuple{N,CuArray}
    ) where {N}
        return init_passive_markers(CUDABackend, coords)
    end

    function JustPIC._2D.advection!(
        particles::PassiveMarkers{CUDABackend},
        method::AbstractAdvectionIntegrator,
        V::NTuple{N,CuArray},
        grid_vxi,
        dt,
    ) where {N}
        return advection!(particles, method, V, grid_vxi, dt)
    end

    function JustPIC._2D.grid2particle!(Fp, xvi, F, particles::PassiveMarkers{CUDABackend})
        grid2particle!(Fp, xvi, F, particles)
        return nothing
    end

    function JustPIC._2D.grid2particle!(
        Fp::NTuple{N,CuArray},
        xvi,
        F::NTuple{N,CuArray},
        particles::PassiveMarkers{CUDABackend},
    ) where {N}
        grid2particle!(Fp, xvi, F, particles)
        return nothing
    end

    function JustPIC._2D.particle2grid!(
        F, Fp, buffer, xi, particles::PassiveMarkers{CUDABackend}
    )
        particle2grid!(F, Fp, buffer, xi, particles)
        return nothing
    end

    # Phase ratio kernels

    function JustPIC._2D.update_phase_ratios!(
        phase_ratios::JustPIC.PhaseRatios{CUDABackend,T}, particles, xci, xvi, phases
    ) where {T<:AbstractMatrix}
        phase_ratios_center!(phase_ratios, particles, xci, phases)
        phase_ratios_vertex!(phase_ratios, particles, xvi, phases)
        # velocity nodes
        phase_ratios_face!(phase_ratios.Vx, particles, xci, phases, :x)
        phase_ratios_face!(phase_ratios.Vy, particles, xci, phases, :y)
        return nothing
    end

    function JustPIC._2D.PhaseRatios(
        ::Type{CUDABackend}, nphases::Integer, ni::NTuple{N,Integer}
    ) where {N}
        return JustPIC._2D.PhaseRatios(Float64, CUDABackend, nphases, ni)
    end

    function JustPIC._2D.PhaseRatios(
        ::Type{T}, ::Type{CUDABackend}, nphases::Integer, ni::NTuple{2,Integer}
    ) where {T}
        nx, ny = ni

        center = cell_array(zero(T), (nphases,), ni)
        vertex = cell_array(zero(T), (nphases,), ni .+ 1)
        Vx = cell_array(zero(T), (nphases,), (nx + 1, ny))
        Vy = cell_array(zero(T), (nphases,), (nx, ny + 1))
        dummy = cell_array(zero(T), (nphases,), (1, 1)) # because it cant be a Union{T, Nothing} type on the GPU....

        return JustPIC.PhaseRatios(
            CUDABackend, center, vertex, Vx, Vy, dummy, dummy, dummy, dummy
        )
    end

    function JustPIC._2D.phase_ratios_center!(
        phase_ratios::JustPIC.PhaseRatios{CUDABackend}, particles, xci, phases
    )
        ni = size(phases)
        di = compute_dx(xci)

        @parallel (@idx ni) phase_ratios_center_kernel!(
            phase_ratios.center, particles.coords, xci, di, phases
        )
        return nothing
    end

    function JustPIC._2D.phase_ratios_vertex!(
        phase_ratios::JustPIC.PhaseRatios{CUDABackend}, particles, xvi, phases
    )
        ni = size(phases) .+ 1
        di = compute_dx(xvi)

        @parallel (@idx ni) phase_ratios_vertex_kernel!(
            phase_ratios.vertex, particles.coords, xvi, di, phases
        )
        return nothing
    end

    function JustPIC._2D.phase_ratios_midpoint!(
        phase_midpoint, particles::Particles{CUDABackend}, xci::NTuple{N}, phases, dimension
    ) where {N}
        phase_ratios_midpoint!(phase_midpoint, particles, xci, phases, dimension)
        return nothing
    end
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

    # halo update
    function JustPIC._3D.update_cell_halo!(
        x::Vararg{CellArray{S,N,D,CuArray{T,nD}},NA}
    ) where {NA,S,N,D,T,nD}
        return update_cell_halo!(x...)
    end
    function JustPIC._3D.update_cell_halo!(
        x::Vararg{CellArray{S,N,D,CuArray{T,nD,B}},NA}
    ) where {NA,S,N,D,T,nD,B}
        return update_cell_halo!(x...)
    end

    # Conversions
    function JustPIC._3D.Particles(
        coords,
        index::CellArray{StaticArraysCore.SVector{N1,Bool},3,0,CuArray{Bool,N2}},
        nxcell,
        max_xcell,
        min_xcell,
        np,
    ) where {N1,N2}
        return Particles(CUDABackend, coords, index, nxcell, max_xcell, min_xcell, np)
    end

    function JustPIC._3D.Particles(
        coords,
        index::CellArray{StaticArraysCore.SVector{N1,Bool},3,0,CuArray{Bool,N2,B}},
        nxcell,
        max_xcell,
        min_xcell,
        np,
    ) where {B,N1,N2}
        return Particles(CUDABackend, coords, index, nxcell, max_xcell, min_xcell, np)
    end

    function JustPIC._3D.SubgridDiffusionCellArrays(
        particles::Particles{CUDABackend}; loc::Symbol=:vertex
    )
        return SubgridDiffusionCellArrays(particles; loc=loc)
    end

    function JustPIC._3D.init_particles(
        ::Type{CUDABackend}, nxcell, max_xcell, min_xcell, x, y, z; buffer=1 - 1e-5
    )
        return init_particles(
            CUDABackend, nxcell, max_xcell, min_xcell, x, y, z; buffer=buffer
        )
    end

    function JustPIC._3D.init_particles(
        ::Type{CUDABackend},
        nxcell,
        max_xcell,
        min_xcell,
        coords::NTuple{3,AbstractArray},
        dxᵢ::NTuple{3,T},
        nᵢ::NTuple{3,I};
        buffer=1 - 1e-5,
    ) where {T,I}
        return init_particles(
            CUDABackend, nxcell, max_xcell, min_xcell, coords, dxᵢ, nᵢ; buffer=buffer
        )
    end

    function JustPIC._3D.advection!(
        particles::Particles{CUDABackend},
        method::AbstractAdvectionIntegrator,
        V,
        grid_vxi::NTuple{N,NTuple{N,T}},
        dt,
    ) where {N,T}
        return advection!(particles, method, V, grid_vxi, dt)
    end

    function JustPIC._3D.advection_LinP!(
        particles::Particles{CUDABackend},
        method::AbstractAdvectionIntegrator,
        V,
        grid_vxi::NTuple{N,NTuple{N,T}},
        dt,
    ) where {N,T}
        return advection_LinP!(particles, method, V, grid_vxi, dt)
    end

    function JustPIC._3D.advection_MQS!(
        particles::Particles{CUDABackend},
        method::AbstractAdvectionIntegrator,
        V,
        grid_vxi::NTuple{N,NTuple{N,T}},
        dt,
    ) where {N,T}
        return advection_MQS!(particles, method, V, grid_vxi, dt)
    end

    function JustPIC._3D.centroid2particle!(
        Fp, xci, F::CuArray, particles::Particles{CUDABackend}
    )
        return centroid2particle!(Fp, xci, F, particles)
    end

    function JustPIC._3D.grid2particle!(
        Fp, xvi, F::CuArray, particles::Particles{CUDABackend}
    )
        return grid2particle!(Fp, xvi, F, particles)
    end

    function JustPIC._3D.particle2centroid!(
        F::CuArray, Fp, xi::NTuple, particles::Particles{CUDABackend}
    )
        return particle2centroid!(F, Fp, xi, particles)
    end

    function JustPIC._3D.particle2grid!(
        F::CuArray, Fp, xi, particles::Particles{CUDABackend}
    )
        return particle2grid!(F, Fp, xi, particles)
    end

    function JustPIC._3D.grid2particle_flip!(Fp, xvi, F::CuArray, F0, particles; α=0.0)
        return grid2particle_flip!(Fp, xvi, F, F0, particles; α=α)
    end

    function JustPIC._3D.inject_particles!(
        particles::Particles{CUDABackend}, args, grid::NTuple{N}
    ) where {N}
        return inject_particles!(particles, args, grid)
    end

    function JustPIC._3D.inject_particles_phase!(
        particles::Particles{CUDABackend}, particles_phases, args, fields, grid::NTuple{N}
    ) where {N}
        inject_particles_phase!(particles::Particles, particles_phases, args, fields, grid)
        return nothing
    end

    function JustPIC._3D.move_particles!(
        particles::Particles{CUDABackend}, grid::NTuple{N}, args
    ) where {N}
        return move_particles!(particles, grid, args)
    end

    function JustPIC._3D.init_cell_arrays(
        particles::Particles{CUDABackend}, V::Val{N}
    ) where {N}
        return init_cell_arrays(particles, V)
    end

    function JustPIC._3D.subgrid_diffusion!(
        pT,
        T_grid,
        ΔT_grid,
        subgrid_arrays,
        particles::Particles{CUDABackend},
        xvi,
        di,
        dt;
        d=1.0,
    )
        subgrid_diffusion!(pT, T_grid, ΔT_grid, subgrid_arrays, particles, xvi, di, dt; d=d)
        return nothing
    end

    function JustPIC._3D.subgrid_diffusion_centroid!(
        pT,
        T_grid,
        ΔT_grid,
        subgrid_arrays,
        particles::Particles{CUDABackend},
        xvi,
        di,
        dt;
        d=1.0,
    )
        subgrid_diffusion_centroid!(
            pT, T_grid, ΔT_grid, subgrid_arrays, particles, xvi, di, dt; d=d
        )
        return nothing
    end

    ## PassiveMarkers

    function JustPIC._3D.init_passive_markers(
        ::Type{CUDABackend}, coords::NTuple{N,CuArray}
    ) where {N}
        return init_passive_markers(CUDABackend, coords)
    end

    function JustPIC._3D.advection!(
        particles::PassiveMarkers{CUDABackend},
        method::AbstractAdvectionIntegrator,
        V::NTuple{N,CuArray},
        grid_vxi,
        dt,
    ) where {N}
        return advection!(particles, method, V, grid_vxi, dt)
    end

    function JustPIC._3D.grid2particle!(Fp, xvi, F, particles::PassiveMarkers{CUDABackend})
        grid2particle!(Fp, xvi, F, particles)
        return nothing
    end

    function JustPIC._3D.grid2particle!(
        Fp::NTuple{N,CuArray},
        xvi,
        F::NTuple{N,CuArray},
        particles::PassiveMarkers{CUDABackend},
    ) where {N}
        grid2particle!(Fp, xvi, F, particles)
        return nothing
    end

    # Phase ratio kernels

    function JustPIC._3D.update_phase_ratios!(
        phase_ratios::JustPIC.PhaseRatios{CUDABackend,T}, particles, xci, xvi, phases
    ) where {T<:AbstractArray}
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
        ::Type{CUDABackend}, nphases::Integer, ni::NTuple{N,Integer}
    ) where {N}
        return JustPIC._3D.PhaseRatios(Float64, CUDABackend, nphases, ni)
    end

    function JustPIC._3D.PhaseRatios(
        ::Type{T}, ::Type{CUDABackend}, nphases::Integer, ni::NTuple{3,Integer}
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

        return JustPIC.PhaseRatios(CUDABackend, center, vertex, Vx, Vy, Vz, yz, xz, xy)
    end

    function JustPIC._3D.phase_ratios_center!(
        phase_ratios::JustPIC.PhaseRatios{CUDABackend}, particles, xci, phases
    )
        ni = size(phases)
        di = compute_dx(xci)

        @parallel (@idx ni) phase_ratios_center_kernel!(
            phase_ratios.center, particles.coords, xci, di, phases
        )
        return nothing
    end

    function JustPIC._3D.phase_ratios_vertex!(
        phase_ratios::JustPIC.PhaseRatios{CUDABackend}, particles, xvi, phases
    )
        ni = size(phases) .+ 1
        di = compute_dx(xvi)

        @parallel (@idx ni) phase_ratios_vertex_kernel!(
            phase_ratios.vertex, particles.coords, xvi, di, phases
        )
        return nothing
    end

    function JustPIC._3D.phase_ratios_midpoint!(
        phase_midpoint, particles::Particles{CUDABackend}, xci::NTuple{N}, phases, dimension
    ) where {N}
        phase_ratios_midpoint!(phase_midpoint, particles, xci, phases, dimension)
        return nothing
    end
end

end # module

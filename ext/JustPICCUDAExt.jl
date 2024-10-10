module JustPICCUDAExt

using CUDA
using JustPIC, CellArrays, StaticArrays

JustPIC.TA(::Type{CUDABackend}) = CuArray

CuCellArray(::Type{T}, ::UndefInitializer, dims::NTuple{N,Int}) where {T<:CellArrays.Cell,N} = CellArrays.CellArray{T,N,0,CUDA.CuArray{eltype(T),3}}(undef, dims)
CuCellArray(::Type{T}, ::UndefInitializer, dims::Int...) where {T<:CellArrays.Cell} = CuCellArray(T, undef, dims)

function CUDA.CuArray(::Type{T}, particles::JustPIC.Particles) where {T<:Number}
    (; coords, index, nxcell, max_xcell, min_xcell, np) = particles
    coords_gpu = ntuple(i->CuArray(T, coords[i]), Val(length(coords))) 
    return Particles(CUDABackend, coords_gpu, CuArray(Bool, index), nxcell, max_xcell, min_xcell, np)
end

function CUDA.CuArray(::Type{T}, phase_ratios::JustPIC.PhaseRatios) where {T<:Number}
    (; vertex, center) = phase_ratios
    return JustPIC.PhaseRatios(CUDABackend, CuArray(T, center), CuArray(T, vertex))
end

function CUDA.CuArray(particles::JustPIC.Particles)
    (; coords, index, nxcell, max_xcell, min_xcell, np) = particles
    coords_gpu = ntuple(i->CuArray(coords[i]), Val(length(coords))) 
    return Particles(CUDABackend, coords_gpu, CuArray(index), nxcell, max_xcell, min_xcell, np)
end

function CUDA.CuArray(phase_ratios::JustPIC.PhaseRatios)
    (; vertex, center) = phase_ratios
    return JustPIC.PhaseRatios(CUDABackend, CuArray(center), CuArray(vertex))
end

function CUDA.CuArray(::Type{T}, CA::CellArray) where {T<:Number}
    ni      = size(CA)
    # Array initializations
    T_SArray = eltype(CA)
    CA_CUDA = CuCellArray(SVector{length(T_SArray), T}, undef, ni...)
    # copy data to the CUDA CellArray
    tmp = if size(CA.data) != size(CA_CUDA.data)
        CuArray(permutedims(CA.data, (3, 2, 1)))
    else
        CuArray(CA.data)
    end
    copyto!(CA_CUDA.data, tmp)
    return CA_CUDA
end

CUDA.CuArray(particles::JustPIC.Particles{CUDABackend})      = particles
CUDA.CuArray(phase_ratios::JustPIC.PhaseRatios{CUDABackend}) = phase_ratios
CUDA.CuArray(CA::CellArray)                                  = CUDA.CuArray(eltype(eltype(CA)), CA)

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

    function JustPIC._2D.SubgridDiffusionCellArrays(particles::Particles{CUDABackend})
        return SubgridDiffusionCellArrays(particles)
    end

    function JustPIC._2D.init_particles(
        ::Type{CUDABackend}, nxcell, max_xcell, min_xcell, x, y
    )
        return init_particles(CUDABackend, nxcell, max_xcell, min_xcell, x, y)
    end

    function JustPIC._2D.init_particles(
        ::Type{CUDABackend},
        nxcell,
        max_xcell,
        min_xcell,
        coords::NTuple{2,AbstractArray},
        dxᵢ::NTuple{2,T},
        nᵢ::NTuple{2,I},
    ) where {T,I}
        return init_particles(CUDABackend, nxcell, max_xcell, min_xcell, coords, dxᵢ, nᵢ)
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

    function JustPIC._2D.inject_particles!(particles::Particles{CUDABackend}, args, grid::NTuple{N}) where N
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
        subgrid_diffusion_centroid!(pT, T_grid, ΔT_grid, subgrid_arrays, particles, xvi, di, dt; d=d)
        return nothing
    end

    ## MakerChain

    function JustPIC._2D.advect_markerchain!(
        chain::MarkerChain{CUDABackend},
        method::AbstractAdvectionIntegrator,
        V,
        grid_vxi,
        dt,
    )
        return advect_markerchain!(chain, method, V, grid_vxi, dt)
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

    function JustPIC._2D.update_phase_ratios!(phase_ratios::JustPIC.PhaseRatios{CUDABackend}, particles, xci, xvi, phases)
        phase_ratios_center!(phase_ratios, particles, xci, phases)
        phase_ratios_vertex!(phase_ratios, particles, xvi, phases)
        return nothing
    end

    function JustPIC._2D.PhaseRatios(
        ::Type{CUDABackend}, nphases::Integer, ni::NTuple{N,Integer}
    ) where {N}
        return JustPIC._2D.PhaseRatios(Float64, CUDABackend, nphases, ni)
    end

    function JustPIC._2D.PhaseRatios(
        ::Type{T}, ::Type{CUDABackend}, nphases::Integer, ni::NTuple{N,Integer}
    ) where {N,T}
        center = cell_array(0.0, (nphases,), ni)
        vertex = cell_array(0.0, (nphases,), ni .+ 1)

        return JustPIC.PhaseRatios(CUDABackend, center, vertex)
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

    function JustPIC._3D.SubgridDiffusionCellArrays(particles::Particles{CUDABackend})
        return SubgridDiffusionCellArrays(particles)
    end

    function JustPIC._3D.init_particles(
        ::Type{CUDABackend}, nxcell, max_xcell, min_xcell, x, y, z
    )
        return init_particles(CUDABackend, nxcell, max_xcell, min_xcell, x, y, z)
    end

    function JustPIC._3D.init_particles(
        ::Type{CUDABackend},
        nxcell,
        max_xcell,
        min_xcell,
        coords::NTuple{3,AbstractArray},
        dxᵢ::NTuple{3,T},
        nᵢ::NTuple{3,I},
    ) where {T,I}
        return init_particles(CUDABackend, nxcell, max_xcell, min_xcell, coords, dxᵢ, nᵢ)
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

    function JustPIC._3D.inject_particles!(particles::Particles{CUDABackend}, args, grid::NTuple{N}) where N
        return inject_particles!(particles, args, grid)
    end

    function JustPIC._3D.inject_particles_phase!(
        particles::Particles{CUDABackend}, particles_phases, args, fields, grid::NTuple{N}
    ) where N
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
        subgrid_diffusion_centroid!(pT, T_grid, ΔT_grid, subgrid_arrays, particles, xvi, di, dt; d=d)
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

    function JustPIC._3D.update_phase_ratios!(phase_ratios::JustPIC.PhaseRatios{CUDABackend}, particles, xci, xvi, phases)
        phase_ratios_center!(phase_ratios, particles, xci, phases)
        phase_ratios_vertex!(phase_ratios, particles, xvi, phases)
        return nothing
    end
    
    function JustPIC._3D.PhaseRatios(
        ::Type{CUDABackend}, nphases::Integer, ni::NTuple{N,Integer}
    ) where {N}
        return JustPIC._3D.PhaseRatios(Float64, CUDABackend, nphases, ni)
    end

    function JustPIC._3D.PhaseRatios(
        ::Type{T}, ::Type{CUDABackend}, nphases::Integer, ni::NTuple{N,Integer}
    ) where {N,T}
        center = cell_array(0.0, (nphases,), ni)
        vertex = cell_array(0.0, (nphases,), ni .+ 1)

        return JustPIC.PhaseRatios(CUDABackend, center, vertex)
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
end

end # module


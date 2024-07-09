module JustPICAMDGPUExt

using JustPIC
using AMDGPU
JustPIC.TA(::Type{JustPIC.AMDGPUBackend}) = ROCArray

module _2D
    using ImplicitGlobalGrid
    using MPI: MPI
    using MuladdMacro
    using ParallelStencil
    using CellArrays
    using StaticArrays
    using AMDGPU
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

    function JustPIC._2D.Particles(coords, index::CellArray{StaticArraysCore.SVector{N1, Bool}, 2, 0, ROCArray{Bool, N2}}, nxcell, max_xcell, min_xcell, np) where {N1,N2}
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

    function JustPIC._2D.move_particles!(particles::Particles{AMDGPUBackend}, grid, args)
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

end

module _3D
    using ImplicitGlobalGrid
    using MPI: MPI
    using MuladdMacro
    using ParallelStencil
    using CellArrays
    using StaticArrays
    using AMDGPU
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

    function JustPIC._3D.Particles(coords, index::CellArray{StaticArraysCore.SVector{N1, Bool}, 3, 0, ROCArray{Bool, N2}}, nxcell, max_xcell, min_xcell, np) where {N1,N2}
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

    function JustPIC._3D.move_particles!(particles::Particles{AMDGPUBackend}, grid, args)
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

end

end # module

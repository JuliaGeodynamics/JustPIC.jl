module JustPICCUDAExt

using CUDA
using JustPIC
JustPIC.TA(::Type{CUDABackend}) = CuArray

module _2D
    using ImplicitGlobalGrid
    using MPI: MPI
    using MuladdMacro
    using ParallelStencil
    using CellArrays
    using StaticArrays
    using CUDA
    using JustPIC

    @init_parallel_stencil(CUDA, Float64, 2)

    const ParticlesExt = JustPIC.Particles
    const PassiveMarkersExt = JustPIC.PassiveMarkers

    import JustPIC._2D.CA

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

    function JustPIC._2D.SubgridDiffusionCellArrays(particles::ParticlesExt{CUDABackend})
        return SubgridDiffusionCellArrays(particles)
    end

    function JustPIC._2D.init_particles(
        ::Type{CUDABackend}, nxcell, max_xcell, min_xcell, x, y, dx, dy, nx, ny
    )
        return init_particles(
            CUDABackend, nxcell, max_xcell, min_xcell, (x, y), (dx, dy), (nx, ny)
        )
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

    function JustPIC._2D.advection_RK!(
        particles::ParticlesExt{CUDABackend},
        V,
        grid_vx::NTuple{2,T},
        grid_vy::NTuple{2,T},
        dt,
        α,
    ) where {T}
        return advection_RK!(particles, V, grid_vx, grid_vy, dt, α)
    end

    function JustPIC._2D.centroid2particle!(
        Fp, xci, F::CuArray, particles::ParticlesExt{CUDABackend}
    )
        return centroid2particle!(Fp, xci, F, particles)
    end

    function JustPIC._2D.grid2particle!(
        Fp, xvi, F::CuArray, particles::ParticlesExt{CUDABackend}
    )
        return grid2particle!(Fp, xvi, F, particles)
    end

    function JustPIC._2D.particle2grid_centroid!(
        F::CuArray, Fp, xi, particles::ParticlesExt{CUDABackend}
    )
        return particle2grid_centroid!(F, Fp, xi, particles)
    end

    function JustPIC._2D.particle2grid!(F::CuArray, Fp, xi, particles)
        return particle2grid!(F, Fp, xi, particles)
    end

    function JustPIC._2D.grid2particle_flip!(Fp, xvi, F::CuArray, F0, particles; α=0.0)
        return grid2particle_flip!(Fp, xvi, F, F0, particles; α=α)
    end

    function JustPIC._2D.check_injection(particles::ParticlesExt{CUDABackend})
        return check_injection(particles)
    end

    function JustPIC._2D.inject_particles!(
        particles::ParticlesExt{CUDABackend}, args, fields, grid
    )
        return inject_particles!(particles, args, fields, grid)
    end

    function JustPIC._2D.inject_particles_phase!(
        particles::ParticlesExt{CUDABackend}, particles_phases, args, fields, grid
    )
        inject_particles_phase!(particles::Particles, particles_phases, args, fields, grid)
        return nothing
    end

    function JustPIC._2D.shuffle_particles!(
        particles::ParticlesExt{CUDABackend}, args::Vararg{Any,N}
    ) where {N}
        return shuffle_particles!(particles, args...)
    end

    function JustPIC._2D.move_particles!(particles::ParticlesExt{CUDABackend}, grid, args)
        return shuffle_particles!(particles, grid, args)
    end

    function JustPIC._2D.init_cell_arrays(
        particles::ParticlesExt{CUDABackend}, V::Val{N}
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
    ## MakerChain

    function JustPIC._2D.advect_markerchain!(
        chain::MarkerChain{CUDABackend}, V, grid_vx, grid_vy, dt
    )
        return advect_markerchain!(chain, V, grid_vx, grid_vy, dt)
    end

    ## PassiveMarkers

    function JustPIC._2D.init_passive_markers(
        ::Type{CUDABackend}, coords::NTuple{N,CuArray}
    ) where {N}
        return init_passive_markers(CUDABackend, coords)
    end

    function JustPIC._2D.advect_passive_markers!(
        particles::PassiveMarkersExt{CUDABackend},
        V::NTuple{N,CuArray},
        grid_vx,
        grid_vy,
        dt;
        α::Float64=2 / 3,
    ) where {N}
        return advect_passive_markers!(particles, V, grid_vx, grid_vy, dt; α=α)
    end

    function JustPIC._2D.grid2particle!(
        Fp, xvi, F, particles::PassiveMarkersExt{CUDABackend}
    )
        grid2particle!(Fp, xvi, F, particles)
        return nothing
    end

    function JustPIC._2D.grid2particle!(
        Fp::NTuple{N,CuArray},
        xvi,
        F::NTuple{N,CuArray},
        particles::PassiveMarkersExt{CUDABackend},
    ) where {N}
        grid2particle!(Fp, xvi, F, particles)
        return nothing
    end

    function JustPIC._2D.particle2grid!(
        F, Fp, buffer, xi, particles::PassiveMarkersExt{CUDABackend}
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
    using CUDA
    using JustPIC

    @init_parallel_stencil(CUDA, Float64, 3)

    macro myatomic(expr)
        return esc(
            quote
                CUDA.@atomic $expr
            end,
        )
    end

    const ParticlesExt = JustPIC.Particles
    const PassiveMarkersExt = JustPIC.PassiveMarkers

    function JustPIC._3D.CA(::Type{CUDABackend}, dims; eltype=Float64)
        return CuCellArray{eltype}(undef, dims)
    end

    include(joinpath(@__DIR__, "../src/common.jl"))

    include(joinpath(@__DIR__, "../src/CUDAExt/CellArrays.jl"))

    function JustPIC._3D.SubgridDiffusionCellArrays(particles::ParticlesExt{CUDABackend})
        return SubgridDiffusionCellArrays(particles)
    end

    function JustPIC._3D.init_particles(
        ::Type{CUDABackend}, nxcell, max_xcell, min_xcell, x, y, z, dx, dy, dz, nx, ny, nz
    )
        return init_particles(
            CUDABackend, nxcell, max_xcell, min_xcell, (x, y, z), (dx, dy, dz), (nx, ny, nz)
        )
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

    function JustPIC._3D.advection_RK!(
        particles::ParticlesExt{CUDABackend},
        V,
        grid_vx::NTuple{3,T},
        grid_vy::NTuple{3,T},
        grid_vz::NTuple{3,T},
        dt,
        α,
    ) where {T}
        return advection_RK!(particles, V, grid_vx, grid_vy, grid_vz, dt, α)
    end

    function JustPIC._3D.centroid2particle!(
        Fp, xci, F::CuArray, particles::ParticlesExt{CUDABackend}
    )
        return centroid2particle!(Fp, xci, F, particles)
    end

    function JustPIC._3D.grid2particle!(
        Fp, xvi, F::CuArray, particles::ParticlesExt{CUDABackend}
    )
        return grid2particle!(Fp, xvi, F, particles)
    end

    function JustPIC._3D.particle2grid_centroid!(
        F::CuArray, Fp, xi, particles::ParticlesExt{CUDABackend}
    )
        return particle2grid_centroid!(F, Fp, xi, particles)
    end

    function JustPIC._3D.particle2grid!(
        F::CuArray, Fp, xi, particles::ParticlesExt{CUDABackend}
    )
        return particle2grid!(F, Fp, xi, particles)
    end

    function JustPIC._3D.grid2particle_flip!(Fp, xvi, F::CuArray, F0, particles; α=0.0)
        return grid2particle_flip!(Fp, xvi, F, F0, particles; α=α)
    end

    function JustPIC._3D.check_injection(particles::ParticlesExt{CUDABackend})
        return check_injection(particles)
    end

    function JustPIC._3D.inject_particles!(
        particles::ParticlesExt{CUDABackend}, args, fields, grid
    )
        return inject_particles!(particles, args, fields, grid)
    end

    function JustPIC._3D.inject_particles_phase!(
        particles::ParticlesExt{CUDABackend}, particles_phases, args, fields, grid
    )
        inject_particles_phase!(particles::Particles, particles_phases, args, fields, grid)
        return nothing
    end

    function JustPIC._3D.shuffle_particles!(
        particles::ParticlesExt{CUDABackend}, args::Vararg{Any,N}
    ) where {N}
        return shuffle_particles!(particles, args...)
    end

    function JustPIC._3D.move_particles!(particles::ParticlesExt{CUDABackend}, grid, args)
        return shuffle_particles!(particles, grid, args)
    end

    function JustPIC._3D.init_cell_arrays(
        particles::ParticlesExt{CUDABackend}, V::Val{N}
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

    ## PassiveMarkers

    function JustPIC._3D.init_passive_markers(
        ::Type{CUDABackend}, coords::NTuple{N,CuArray}
    ) where {N}
        return init_passive_markers(CUDABackend, coords)
    end

    function JustPIC._3D.advect_passive_markers!(
        particles::PassiveMarkersExt{CUDABackend},
        V::NTuple{N,CuArray},
        grid_vx,
        grid_vy,
        grid_vz,
        dt;
        α::Float64=2 / 3,
    ) where {N}
        return advect_passive_markers!(particles, V, grid_vx, grid_vy, grid_vz, dt; α=α)
    end

    function JustPIC._3D.grid2particle!(
        Fp, xvi, F, particles::PassiveMarkersExt{CUDABackend}
    )
        grid2particle!(Fp, xvi, F, particles)
        return nothing
    end

    function JustPIC._3D.grid2particle!(
        Fp::NTuple{N,CuArray},
        xvi,
        F::NTuple{N,CuArray},
        particles::PassiveMarkersExt{CUDABackend},
    ) where {N}
        grid2particle!(Fp, xvi, F, particles)
        return nothing
    end

end

end # module

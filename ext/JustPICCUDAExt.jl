module JustPICCUDAExt

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

    __precompile__(false)

    const ParticlesExt = JustPIC.Particles

    JustPIC._2D.TA(::Type{CUDABackend}) = CuArray

    include(joinpath(@__DIR__, "../src/common.jl"))

    include(joinpath(@__DIR__, "../src/CUDAExt/CellArrays.jl"))

    function JustPIC._2D.init_particles(::Type{CUDABackend}, args::Vararg{Any,N}) where {N}
        return init_particles(CUDABackend, args...)
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

    function JustPIC._2D.centroid2particle!(Fp, xci, F::CuArray, particles::ParticlesExt{CUDABackend})
        return centroid2particle!(Fp, xci, F, particles)
    end

    function JustPIC._2D.grid2particle!(Fp, xvi, F::CuArray, particles::ParticlesExt{CUDABackend})
        return grid2particle!(Fp, xvi, F, particles)
    end

    function JustPIC._2D.particle2grid_centroid!(F::CuArray, Fp, xi, particles::ParticlesExt{CUDABackend})
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
        return nothing4
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

    ## MakerChain

    function JustPIC._2D.advect_markerchain!(
        chain::ParticlesExt{CUDABackend}, V, grid_vx, grid_vy, dt
    )
        return advect_markerchain!(chain, V, grid_vx, grid_vy, dt)
    end

    ## PassiveMarkers

    JustPIC._2D.init_passive_markers(::Type{CUDABackend}, coords)= init_passive_markers(CUDABackend, coords)

    function JustPIC._2D.advect_passive_markers!(
        particles::ParticlesExt{CUDABackend}, V, grid_vx, grid_vy, dt, α,
    )
        return advect_passive_markers!(particles, V, grid_vx, grid_vy, dt, α)
    end

    function JustPIC._2D.grid2particle!(Fp, xvi, F, particles::ParticlesExt{CUDABackend}) 
        grid2particle!(Fp, xvi, F, particles)
        return nothing
    end

    function JustPIC._2D.grid2particle!(Fp::NTuple{N, CuArray}, xvi, F::NTuple{N, CuArray}, particles::ParticlesExt{CUDABackend}) where N
        grid2particle!(Fp, xvi, F, particles)
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

    __precompile__(false)

    const ParticlesExt = JustPIC.Particles

    JustPIC._3D.TA(::Type{CUDABackend}) = CuArray

    include(joinpath(@__DIR__, "../src/common.jl"))

    include(joinpath(@__DIR__, "../src/CUDAExt/CellArrays.jl"))

    function JustPIC._3D.init_particles(::Type{CUDABackend}, args::Vararg{Any,N}) where {N}
        return init_particles(CUDABackend, args...)
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

    function JustPIC._3D.centroid2particle!(Fp, xci, F::CuArray, particles::ParticlesExt{CUDABackend})
        return centroid2particle!(Fp, xci, F, particles)
    end

    function JustPIC._3D.grid2particle!(Fp, xvi, F::CuArray, particles::ParticlesExt{CUDABackend})
        return grid2particle!(Fp, xvi, F, particles)
    end

    function JustPIC._2D.particle2grid_centroid!(F::CuArray, Fp, xi, particles::ParticlesExt{CUDABackend})
        return particle2grid_centroid!(F, Fp, xi, particles)
    end

    function JustPIC._3D.particle2grid!(F::CuArray, Fp, xi, particles::ParticlesExt{CUDABackend})
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
        return nothing4
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

    ## PassiveMarkers

    JustPIC._3D.init_passive_markers(::Type{CUDABackend}, coords)= init_passive_markers(CUDABackend, coords)

    function JustPIC._3D.advect_passive_markers!(
        particles::ParticlesExt{CUDABackend}, V, grid_vx, grid_vy, grid_vz, dt, α,
    )
        return advect_passive_markers!(particles, V, grid_vx, grid_vy, grid_vz, dt, α)
    end

    function JustPIC._3D.grid2particle!(Fp, xvi, F, particles::ParticlesExt{CUDABackend}) 
        grid2particle!(Fp, xvi, F, particles)
        return nothing
    end

    function JustPIC._3D.grid2particle!(Fp::NTuple{N, CuArray}, xvi, F::NTuple{N, CuArray}, particles::ParticlesExt{CUDABackend}) where N
        grid2particle!(Fp, xvi, F, particles)
        return nothing
    end

end

end # module

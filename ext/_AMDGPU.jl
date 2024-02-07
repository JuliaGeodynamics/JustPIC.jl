module _AMDGPU

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

    __precompile__(false)

    const ParticlesExt = JustPIC.Particles

    JustPIC._2D.TA(::Type{AMDGPUBackend}) = ROCArray

    include(joinpath(@__DIR__, "../src/common.jl"))

    include(joinpath(@__DIR__, "../src/AMDGPUExt/CellArrays.jl"))

    function JustPIC._2D.init_particles(
        ::Type{AMDGPUBackend}, args::Vararg{Any,N}
    ) where {N}
        return init_particles(AMDGPUBackend, args...)
    end

    function JustPIC._2D.advection_RK!(
        particles::ParticlesExt{AMDGPUBackend},
        V,
        grid_vx::NTuple{2,T},
        grid_vy::NTuple{2,T},
        dt,
        α,
    ) where {T}
        return advection_RK!(particles, V, grid_vx, grid_vy, dt, α)
    end

    function JustPIC._2D.centroid2particle!(
        Fp, xci, F::ROCArray, args::Vararg{Any,N}
    ) where {N}
        return centroid2particle!(Fp, xci, F, args...)
    end

    function JustPIC._2D.grid2particle!(Fp, xvi, F::ROCArray, args::Vararg{Any,N}) where {N}
        return grid2particle!(Fp, xvi, F, args...)
    end

    function JustPIC._2D.particle2grid_centroid!(F::ROCArray, args::Vararg{Any,N}) where {N}
        return particle2grid_centroid!(F, args...)
    end

    function JustPIC._2D.particle2grid!(F::ROCArray, args::Vararg{Any,N}) where {N}
        return particle2grid!(F, args...)
    end

    function JustPIC._2D.grid2particle_flip!(Fp, xvi, F::ROCArray, F0, particles; α=0.0)
        return grid2particle_flip!(Fp, xvi, F, F0, particles; α=α)
    end

    function JustPIC._2D.check_injection(particles::ParticlesExt{AMDGPUBackend})
        return check_injection(particles)
    end

    function JustPIC._2D.inject_particles!(
        particles::ParticlesExt{AMDGPUBackend}, args, fields, grid
    )
        return inject_particles!(particles, args, fields, grid)
    end

    function JustPIC._2D.shuffle_particles!(
        particles::ParticlesExt{AMDGPUBackend}, args::Vararg{Any,N}
    ) where {N}
        return shuffle_particles!(particles, args...)
    end

    function JustPIC._2D.move_particles!(
        particles::ParticlesExt{AMDGPUBackend}, args::Vararg{Any,N}
    ) where {N}
        return move_particles!(particles, args...)
    end

    function JustPIC._2D.init_cell_arrays(
        particles::ParticlesExt{AMDGPUBackend}, V::Val{N}
    ) where {N}
        return init_cell_arrays(particles, V)
    end

    ## MakerChain

    function JustPIC._2D.advect_markerchain!(
        chain::ParticlesExt{AMDGPUBackend}, V, grid_vx, grid_vy, dt
    )
        return advect_markerchain!(chain, V, grid_vx, grid_vy, dt)
    end

    ## PassiveMarkers

    JustPIC._2D.init_passive_markers(::Type{AMDGPUBackend}, coords)= init_passive_markers(AMDGPUBackend, coords)

    function JustPIC._2D.advect_passive_markers!(
        particles::ParticlesExt{AMDGPUBackend}, V, grid_vx, grid_vy, dt, α,
    )
        return advect_passive_markers!(particles, V, grid_vx, grid_vy, dt, α)
    end

    function JustPIC._2D.grid2particle!(Fp, xvi, F, particles::ParticlesExt{AMDGPUBackend}) 
        grid2particle!(Fp, xvi, F, particles)
        return nothing
    end

    function JustPIC._2D.grid2particle!(Fp::NTuple{N, CuArray}, xvi, F::NTuple{N, CuArray}, particles::ParticlesExt{AMDGPUBackend}) where N
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
    using AMDGPU
    using JustPIC

    @init_parallel_stencil(AMDGPU, Float64, 3)

    __precompile__(false)

    const ParticlesExt = JustPIC.Particles

    JustPIC._3D.TA(::Type{AMDGPUBackend}) = ROCArray

    include(joinpath(@__DIR__, "../src/common.jl"))

    include(joinpath(@__DIR__, "../src/AMDGPUExt/CellArrays.jl"))

    function JustPIC._3D.init_particles(
        ::Type{AMDGPUBackend}, args::Vararg{Any,N}
    ) where {N}
        return init_particles(AMDGPUBackend, args...)
    end

    function JustPIC._3D.advection_RK!(
        particles::ParticlesExt{AMDGPUBackend},
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
        Fp, xci, F::ROCArray, args::Vararg{Any,N}
    ) where {N}
        return centroid2particle!(Fp, xci, F, args...)
    end

    function JustPIC._3D.grid2particle!(Fp, xvi, F::ROCArray, args::Vararg{Any,N}) where {N}
        return grid2particle!(Fp, xvi, F, args...)
    end

    function JustPIC._3D.particle2grid_centroid!(F::ROCArray, args::Vararg{Any,N}) where {N}
        return particle2grid_centroid!(F, args...)
    end

    function JustPIC._3D.particle2grid!(F::ROCArray, args::Vararg{Any,N}) where {N}
        return particle2grid!(F, args...)
    end

    function JustPIC._3D.grid2particle_flip!(Fp, xvi, F::ROCArray, F0, particles; α=0.0)
        return grid2particle_flip!(Fp, xvi, F, F0, particles; α=α)
    end

    function JustPIC._3D.check_injection(particles::ParticlesExt{AMDGPUBackend})
        return check_injection(particles)
    end

    function JustPIC._3D.inject_particles!(
        particles::ParticlesExt{AMDGPUBackend}, args, fields, grid
    )
        return inject_particles!(particles, args, fields, grid)
    end

    function JustPIC._3D.shuffle_particles!(
        particles::ParticlesExt{AMDGPUBackend}, args::Vararg{Any,N}
    ) where {N}
        return shuffle_particles!(particles, args...)
    end

    function JustPIC._3D.move_particles!(
        particles::ParticlesExt{AMDGPUBackend}, args::Vararg{Any,N}
    ) where {N}
        return move_particles!(particles, args...)
    end

    function JustPIC._3D.init_cell_arrays(
        particles::ParticlesExt{AMDGPUBackend}, V::Val{N}
    ) where {N}
        return init_cell_arrays(particles, V)
    end
    
    ## PassiveMarkers

    JustPIC._3D.init_passive_markers(::Type{AMDGPUBackend}, coords)= init_passive_markers(AMDGPUBackend, coords)

    function JustPIC._3D.advect_passive_markers!(
        particles::ParticlesExt{AMDGPUBackend}, V, grid_vx, grid_vy, grid_vz, dt, α,
    )
        return advect_passive_markers!(particles, V, grid_vx, grid_vy, grid_vz, dt, α)
    end

    function JustPIC._3D.grid2particle!(Fp, xvi, F, particles::ParticlesExt{AMDGPUBackend}) 
        grid2particle!(Fp, xvi, F, particles)
        return nothing
    end

    function JustPIC._3D.grid2particle!(Fp::NTuple{N, CuArray}, xvi, F::NTuple{N, CuArray}, particles::ParticlesExt{AMDGPUBackend}) where N
        grid2particle!(Fp, xvi, F, particles)
        return nothing
    end

end

end # module

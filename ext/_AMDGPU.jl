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
    function JustPIC._2D.grid2particle_flip!(
        Fp, xvi, F::ROCArray, F0, particle_coords; α=0.0
    )
        return grid2particle_flip!(Fp, xvi, F, F0, particle_coords; α=α)
    end
    function JustPIC._2D.check_injection(particles::ParticlesExt{AMDGPUBackend})
        return check_injection(particles)
    end
    function JustPIC._2D.inject_particles!(
        particles::ParticlesExt{AMDGPUBackend}, args::Vararg{Any,N}
    ) where {N}
        return inject_particles!(particles, args...)
    end
    function JustPIC._2D.shuffle_particles!(
        particles::ParticlesExt{AMDGPUBackend}, args::Vararg{Any,N}
    ) where {N}
        return shuffle_particles!(particles, args...)
    end
    function JustPIC._2D.init_cell_arrays(
        particles::ParticlesExt{AMDGPUBackend}, V::Val{N}
    ) where {N}
        return init_cell_arrays(particles, V)
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
        grid_vx::NTuple{2,T},
        grid_vy::NTuple{2,T},
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
    function JustPIC._3D.grid2particle_flip!(
        Fp, xvi, F::ROCArray, F0, particle_coords; α=0.0
    )
        return grid2particle_flip!(Fp, xvi, F, F0, particle_coords; α=α)
    end
    function JustPIC._3D.check_injection(particles::ParticlesExt{AMDGPUBackend})
        return check_injection(particles)
    end
    function JustPIC._3D.inject_particles!(
        particles::ParticlesExt{AMDGPUBackend}, args::Vararg{Any,N}
    ) where {N}
        return inject_particles!(particles, args...)
    end
    function JustPIC._3D.shuffle_particles!(
        particles::ParticlesExt{AMDGPUBackend}, args::Vararg{Any,N}
    ) where {N}
        return shuffle_particles!(particles, args...)
    end
    function JustPIC._3D.init_cell_arrays(
        particles::ParticlesExt{AMDGPUBackend}, V::Val{N}
    ) where {N}
        return init_cell_arrays(particles, V)
    end
end

end # module
module _CUDA

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

    function JustPIC._2D.centroid2particle!(
        Fp, xci, F::CuArray, args::Vararg{Any,N}
    ) where {N}
        return centroid2particle!(Fp, xci, F, args...)
    end
    function JustPIC._2D.grid2particle!(Fp, xvi, F::CuArray, args::Vararg{Any,N}) where {N}
        return grid2particle!(Fp, xvi, F, args...)
    end
    function JustPIC._2D.particle2grid_centroid!(F::CuArray, args::Vararg{Any,N}) where {N}
        return particle2grid_centroid!(F, args...)
    end
    function JustPIC._2D.particle2grid!(F::CuArray, args::Vararg{Any,N}) where {N}
        return particle2grid!(F, args...)
    end
    function JustPIC._2D.grid2particle_flip!(
        Fp, xvi, F::CuArray, F0, particle_coords; α=0.0
    )
        return grid2particle_flip!(Fp, xvi, F, F0, particle_coords; α=α)
    end
    function JustPIC._2D.check_injection(particles::ParticlesExt{CUDABackend})
        return check_injection(particles)
    end
    function JustPIC._2D.inject_particles!(
        particles::ParticlesExt{CUDABackend}, args::Vararg{Any,N}
    ) where {N}
        return inject_particles!(particles, args...)
    end
    function JustPIC._2D.shuffle_particles!(
        particles::ParticlesExt{CUDABackend}, args::Vararg{Any,N}
    ) where {N}
        return shuffle_particles!(particles, args...)
    end
    function JustPIC._2D.init_cell_arrays(
        particles::ParticlesExt{CUDABackend}, V::Val{N}
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
        grid_vx::NTuple{2,T},
        grid_vy::NTuple{2,T},
        grid_vz::NTuple{3,T},
        dt,
        α,
    ) where {T}
        return advection_RK!(particles, V, grid_vx, grid_vy, grid_vz, dt, α)
    end

    function JustPIC._3D.centroid2particle!(
        Fp, xci, F::CuArray, args::Vararg{Any,N}
    ) where {N}
        return centroid2particle!(Fp, xci, F, args...)
    end
    function JustPIC._3D.grid2particle!(Fp, xvi, F::CuArray, args::Vararg{Any,N}) where {N}
        return grid2particle!(Fp, xvi, F, args...)
    end
    function JustPIC._3D.particle2grid_centroid!(F::CuArray, args::Vararg{Any,N}) where {N}
        return particle2grid_centroid!(F, args...)
    end
    function JustPIC._3D.particle2grid!(F::CuArray, args::Vararg{Any,N}) where {N}
        return particle2grid!(F, args...)
    end
    function JustPIC._3D.grid2particle_flip!(
        Fp, xvi, F::CuArray, F0, particle_coords; α=0.0
    )
        return grid2particle_flip!(Fp, xvi, F, F0, particle_coords; α=α)
    end
    function JustPIC._3D.check_injection(particles::ParticlesExt{CUDABackend})
        return check_injection(particles)
    end
    function JustPIC._3D.inject_particles!(
        particles::ParticlesExt{CUDABackend}, args::Vararg{Any,N}
    ) where {N}
        return inject_particles!(particles, args...)
    end
    function JustPIC._3D.shuffle_particles!(
        particles::ParticlesExt{CUDABackend}, args::Vararg{Any,N}
    ) where {N}
        return shuffle_particles!(particles, args...)
    end
    function JustPIC._3D.init_cell_arrays(
        particles::ParticlesExt{CUDABackend}, V::Val{N}
    ) where {N}
        return init_cell_arrays(particles, V)
    end
end

end # module

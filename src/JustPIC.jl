module JustPIC

using ImplicitGlobalGrid
using MPI: MPI
using MuladdMacro
using ParallelStencil
using CUDA
using AMDGPU
using CellArrays
using Preferences

# CUDA.allowscalar(false)
__precompile__(false)

function set_backend(new_backend::String)
    if !(
        new_backend ∈ (
            "Threads_Float64_2D",
            "Threads_Float32_2D",
            "Threads_Float64_3D",
            "Threads_Float32_3D",
            "CUDA_Float32_2D",
            "CUDA_Float64_2D",
            "CUDA_Float64_3D",
            "CUDA_Float32_3D",
            "AMDGPU_Float32_2D",
            "AMDGPU_Float64_2D",
            "AMDGPU_Float64_3D",
            "AMDGPU_Float32_3D",
        )
    )
        throw(ArgumentError("Invalid backend: \"$(new_backend)\""))
    end

    # Set it in our runtime values, as well as saving it to disk
    @set_preferences!("backend" => new_backend)
    @info("New backend set; restart your Julia session for this change to take effect!")
end

const backend = @load_preference("backend", "Threads_Float64_2D")

const TA = if occursin("CUDA", backend)
    CUDA.CuArray
elseif occursin("AMDGPU", backend)
    AMDGPU.ROCArray
else
    Array
end

export backend, set_backend, TA

let
    s = split(backend, "_")
    device = s[1]
    precission = s[2]
    dimension = parse(Int, s[3][1])
    @eval begin
        println("Backend: $backend")
        @init_parallel_stencil($(Symbol(device)), $(Symbol(precission)), $dimension)
    end
end

include("CellArrays/CellArrays.jl")
export @cell, cellnum, cellaxes

include("Utils.jl")
export @range, init_cell_arrays, cell_array, add_ghost_nodes, add_global_ghost_nodes

include("CellArrays/ImplicitGlobalGrid.jl")
export update_cell_halo!

# INTERPOLATION RELATED FILES

include("Interpolations/utils.jl")

include("Interpolations/particle_to_grid.jl")
export particle2grid!

include("Interpolations/particle_to_grid_centroid.jl")
export particle2grid_centroid!

include("Interpolations/grid_to_particle.jl")
export grid2particle!, grid2particle_flip!

include("Interpolations/kernels.jl")
export lerp, bilinear, trilinear

# PARTICLES RELATED FILES

include("Particles/particles.jl")
export Particles

include("Particles/utils.jl")

include("Particles/advection.jl")
export advection_RK!

include("Particles/injection.jl")
export check_injection, inject_particles!, inject_particles_phase!, clean_particles!

include("Particles/shuffle.jl")
export shuffle_particles!

include("Particles/shuffle_periodic.jl")
export shuffle_particles_periodic!

end # module

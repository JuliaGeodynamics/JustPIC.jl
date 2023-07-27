module JustPIC

using ImplicitGlobalGrid
import MPI
using MuladdMacro
using ParallelStencil
using CUDA
using CellArrays
using Preferences

function set_backend(new_backend::String)
    if !(new_backend in ("Threads", "CUDA"))
        throw(ArgumentError("Invalid backend: \"$(new_backend)\""))
    end

    # Set it in our runtime values, as well as saving it to disk
    @set_preferences!("backend" => new_backend)
    @info("New backend set; restart your Julia session for this change to take effect!")
end

const backend = @load_preference("backend", "Threads")

const TA = backend == "CUDA" ? JustPIC.CUDA.CuArray : Array

export backend, set_backend, TA

@eval @init_parallel_stencil($(Symbol(backend)), Float64, 2) 

include("CellArrays/CellArrays.jl")
export @cell, cellnum, cellaxes

include("Utils.jl")
export @range, @idx, init_cell_arrays, cell_array, add_ghost_nodes, add_global_ghost_nodes

include("CellArrays/ImplicitGlobalGrid.jl")
export update_cell_halo!

# INTERPOLATION RELATED FILES

include("Interpolations/utils.jl")

include("Interpolations/particle_to_grid.jl")
export particle2grid!

include("Interpolations/grid_to_particle.jl")
export grid2particle!

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

end # module

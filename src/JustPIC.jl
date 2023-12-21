module JustPIC

using ImplicitGlobalGrid
using MPI: MPI
using MuladdMacro
using ParallelStencil
# using CUDA
# using AMDGPU
using CellArrays
using Preferences
using StaticArrays

__precompile__(false)

include("backend.jl")
export backend, set_backend, TA

include("CellArrays/CellArrays.jl")
export @cell, cellnum, cellaxes

include("utils.jl")
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

include("Interpolations/centroid_to_particle.jl")
export centroid2particle!, centroid2particle_flip!

include("Interpolations/kernels.jl")
export lerp, bilinear, trilinear

# PARTICLES RELATED FILES

include("Particles/particles.jl")
export Particles, ClassicParticles, get_cell, nparticles

include("Particles/utils.jl")

include("Particles/advection.jl")
export advection_RK!

include("Particles/injection.jl")
export check_injection, inject_particles!, inject_particles_phase!, clean_particles!

include("Particles/shuffle.jl")
export shuffle_particles!

include("Particles/move.jl")
export move_particles!

# CLASSIC PIC LAYOUT

include("Classic/advection.jl")
export advection_RK!

include("Classic/grid_to_particle.jl")
export grid2particle_naive! #, grid2particle_flip!

include("Classic/particle_to_grid.jl")
export particle2grid_naive!

end # module

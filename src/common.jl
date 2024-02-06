
include("CellArrays/CellArrays.jl")
export @cell, cellnum, cellaxes

include("Utils.jl")
export @range, cell_array, add_ghost_nodes, add_global_ghost_nodes, doskip

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

include("Particles/particles_utils.jl")
export init_particles, init_cell_arrays

include("Particles/utils.jl")

include("Particles/move.jl")
export move_particles!

include("Particles/advection.jl")
export advection_RK!

include("Particles/injection.jl")
export check_injection, inject_particles!, inject_particles_phase!, clean_particles!

include("Particles/shuffle.jl")
export shuffle_particles!

## MARKER CHAIN RELATED FILES

include("MarkerChain/init.jl")
export init_markerchain

include("MarkerChain/move.jl")
export move_particles!

include("MarkerChain/advection.jl")
export advection_RK!

include("MarkerChain/interp1.jl")

include("MarkerChain/sort.jl")
export sort_chain!

include("MarkerChain/resample.jl")
export resample!

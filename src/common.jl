include("CellArrays/CellArrays.jl")
export @cell, cellnum, cellaxes

include("Utils.jl")
export cell_array, add_ghost_nodes, add_global_ghost_nodes, doskip

include("CellArrays/ImplicitGlobalGrid.jl")
export update_cell_halo!

include("Advection/common.jl")

# INTERPOLATION RELATED FILES

include("Interpolations/utils.jl")

include("Interpolations/particle_to_grid.jl")
export particle2grid!

include("Interpolations/particle_to_grid_centroid.jl")
export particle2centroid!

include("Interpolations/grid_to_particle.jl")
export grid2particle!, grid2particle_flip!

include("Interpolations/centroid_to_particle.jl")
export centroid2particle!

include("Interpolations/ndlerp.jl")

include("Interpolations/MQS.jl")

include("Physics/subgrid_diffusion.jl")
export SubgridDiffusionCellArrays, subgrid_diffusion!, subgrid_diffusion_centroid!

# PARTICLES RELATED FILES

include("Particles/particles_utils.jl")
export init_particles, init_cell_arrays

include("Particles/utils.jl")

# include("Particles/move.jl")
include("Particles/move_safe.jl")
export move_particles!

include("Particles/Advection/Euler.jl")
include("Particles/Advection/RK2.jl")
include("Particles/Advection/RK4.jl")
include("Particles/Advection/advection.jl")
include("Particles/Advection/advection_LinP.jl")
include("Particles/Advection/advection_MQS.jl")
export advection!, advection_LinP!, advection_MQS!

include("Particles/injection.jl")
export inject_particles!, inject_particles_phase!, clean_particles!

include("Particles/forced_injection.jl")
export force_injection!

## MARKER CHAIN RELATED FILES

include("MarkerChain/init.jl")
export init_markerchain, fill_chain_from_chain!, fill_chain_from_vertices!

include("MarkerChain/bilinear_MC.jl")
export compute_topography_vertex!

include("MarkerChain/move.jl")
export move_particles!

include("MarkerChain/interp1.jl")

include("MarkerChain/resample.jl")
export resample!

include("MarkerChain/areas.jl")
# include("MarkerChain/areas0.jl")
export compute_rock_fraction!

include("MarkerChain/Advection/Euler.jl")
include("MarkerChain/Advection/RK2.jl")
include("MarkerChain/Advection/RK4.jl")
include("MarkerChain/Advection/advection.jl")
export advection!, advect_markerchain!

include("MarkerChain/Advection/interp_velocity.jl")
export interpolate_velocity_to_markerchain!

## PASSIVE MARKERS RELATED FILES

include("PassiveMarkers/init.jl")
export init_passive_markers

include("PassiveMarkers/advection.jl")
export advection!

include("PassiveMarkers/grid_to_particle.jl")
export grid2particle!

include("PassiveMarkers/particle_to_grid.jl")
export particle2grid!

include("PhaseRatios/constructors.jl")
export PhaseRatios

include("PhaseRatios/utils.jl")
include("PhaseRatios/centers.jl")
include("PhaseRatios/vertices.jl")
include("PhaseRatios/midpoints.jl")
export update_phase_ratios!,
    phase_ratios_center!, phase_ratios_vertex!, phase_ratios_midpoint!

include("IO/JLD2.jl")
export checkpointing_particles

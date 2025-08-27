using Statistics

function semilagrangian_advection_markerchain!(
        chain::MarkerChain, method::AbstractAdvectionIntegrator, V, grid_vxi, grid, dt;
        max_slope_angle = 45.0
    )

    semilagrangian_advection!(chain, method, V, grid_vxi, grid, dt)

    # Apply LaMEM-style slope limiting
    smooth_slopes!(chain, deg2rad(max_slope_angle))

    # Mass conservation
    chain.h_vertices .+= mean(chain.h_vertices) - mean(chain.h_vertices0)

    # Reconstruct particles from the updated vertices
    reconstruct_chain_from_vertices!(chain)
    # update old nodal topography
    copyto!(chain.h_vertices0, chain.h_vertices)

    return nothing
end

# Two-step Runge-Kutta advection scheme for marker chains
function semilagrangian_advection!(
        chain::MarkerChain,
        method::AbstractAdvectionIntegrator,
        V,
        grid_vxi::NTuple{N, NTuple{N, T}},
        grid,
        dt,
    ) where {N, T}
    (; h_vertices) = chain

    # compute some basic stuff
    ni = length(h_vertices)
    dxi = compute_dx(first(grid_vxi))
    local_limits = inner_limits(grid_vxi)

    # launch parallel advection kernel
    @parallel (@idx ni) semilagrangian_advection_markerchain_kernel!(
        h_vertices, method, V, grid_vxi, grid, local_limits, dxi, dt
    )
    return nothing
end

# DIMENSION AGNOSTIC KERNELS

# ParallelStencil function Runge-Kuttaadvection function for 3D staggered grids
@parallel_indices (i) function semilagrangian_advection_markerchain_kernel!(
        h_vertices,
        method::AbstractAdvectionIntegrator,
        V::NTuple{N, T},
        grid_vxi,
        grid,
        local_limits,
        dxi,
        dt,
    ) where {N, T}

    hᵢ = h_vertices[i]
    pᵢ = grid[1][i], hᵢ
    # backtrack particle position
    _, hᵢ_new = advect_particle_markerchain(method, pᵢ, V, grid_vxi, local_limits, dxi, dt; backtracking = true)
    h_vertices[i] -= (hᵢ_new - h_vertices[i])

    return nothing
end

# function compute_topography_vertex_with_neighbors!(chain::MarkerChain)
#     (; coords, index, cell_vertices, h_vertices) = chain

#     @parallel (1:length(cell_vertices)) _compute_h_vertex_with_neighbors!(
#         h_vertices, coords, index, cell_vertices
#     )

#     return nothing
# end

# @parallel_indices (ivertex) function _compute_h_vertex_with_neighbors!(
#         h_vertices, coords, index, cell_vertices
#     )
#     xcorner = cell_vertices[ivertex]

#     # Find which cell this vertex belongs to
#     I = max(1, min(length(index), ivertex))

#     # Check if we're at the boundaries and handle accordingly
#     if I == 1 || I == length(index)
#         # For boundary vertices, use simpler interpolation
#         x_cell = @cell coords[1][I]
#         y_cell = @cell coords[2][I]
#         h_vertices[ivertex] = interp1D_extremas(xcorner, x_cell, y_cell)
#     else
#         # For interior vertices, use the neighbor-aware interpolation
#         x_cell = @cell coords[1][I]
#         y_cell = @cell coords[2][I]
#         h_vertices[ivertex] = interp1D_inner(xcorner, x_cell, y_cell, coords, I)
#     end

#     return nothing
# end

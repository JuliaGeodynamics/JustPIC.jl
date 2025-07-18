using Statistics

function semilagrangian_advection_markerchain!(
        chain::MarkerChain, method::AbstractAdvectionIntegrator, V, grid_vxi, grid, dt
    )
    copyto!(chain.h_vertices0, chain.h_vertices)
    semilagrangian_advection!(chain, method, V, grid_vxi, grid, dt)
    # correct topo to conserve mass
    chain.h_vertices .+= mean(chain.h_vertices) - mean(chain.h_vertices0)
    # update old nodal topography
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
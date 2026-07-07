using Statistics

"""
    semilagrangian_advection_markerchain!(chain, method, V, grid_vxi, grid, dt; max_slope_angle = 45.0)

Backtrack a marker chain through `V` and update the chain geometry with a
semi-Lagrangian step.

The optional `max_slope_angle` limiter is used while reconstructing the interface
to avoid excessively steep local segments.
"""
function semilagrangian_advection_markerchain!(
        chain::MarkerChain, method::AbstractAdvectionIntegrator, V, grid_vxi, grid, dt;
        max_slope_angle = 45.0
    )

    semilagrangian_advection!(chain, method, V, grid_vxi, grid, dt)

    # Apply LaMEM-style slope limiting
    smooth_slopes!(chain, deg2rad(max_slope_angle))

    # Mass conservation
    chain.h_vertices .-= mean(chain.h_vertices) - mean(chain.h_vertices0)

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

    # recast integrator/timestep/grids to the topography precision so Float32 backends
    # (e.g. Metal) are not silently promoted; `recast_grid` also rebuilds the grid ranges
    # so they are GPU-safe when indexed directly inside the kernel (see advection!)
    Tc = eltype(h_vertices)
    method = set_precision(method, Tc)
    dt = convert(Tc, dt)
    grid_vxi = recast_grid(grid_vxi, Tc)
    grid = recast_grid(grid, Tc)

    # compute some basic stuff
    ni = length(h_vertices)
    dxi = compute_dx(first(grid_vxi))
    local_limits = inner_limits(grid_vxi)

    # launch parallel advection kernel
    launch!(
        ka_backend(h_vertices), semilagrangian_advection_markerchain_kernel!, ni,
        h_vertices, method, V, grid_vxi, grid, local_limits, dxi, dt
    )
    return nothing
end

# DIMENSION AGNOSTIC KERNELS

# Semilagrangian backtracking kernel for staggered grids.
@kernel function semilagrangian_advection_markerchain_kernel!(
        h_vertices,
        method::AbstractAdvectionIntegrator,
        V::NTuple{N, T},
        grid_vxi,
        grid,
        local_limits,
        dxi,
        dt,
    ) where {N, T}
    i = @index(Global)

    hᵢ = h_vertices[i]
    pᵢ = grid[1][i], hᵢ
    # backtrack particle position
    _, hᵢ_new = advect_particle_markerchain(method, pᵢ, V, grid_vxi, local_limits, dxi, dt; backtracking = true)
    h_vertices[i] -= (hᵢ_new - h_vertices[i])
end

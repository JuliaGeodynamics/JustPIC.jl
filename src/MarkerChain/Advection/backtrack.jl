using Statistics

"""
    semilagrangian_advection_markerchain!(chain, method, V, grid_vxi, grid, dt; max_slope_angle = 45.0)

Backtrack a marker chain through `V` and update the chain geometry with a
semi-Lagrangian step.

Unlike [`advect_markerchain!`](@ref), which moves the Lagrangian markers, this scheme
updates the vertex topography directly by backtracking each vertex (via the lower-level
`semilagrangian_advection!`), then limits the local slope to `max_slope_angle` degrees
(via `smooth_slopes!`), restores the mean height for mass conservation, and finally rebuilds
the markers from the updated vertices. It is well suited to steep or strongly sheared
surfaces where marker advection would tangle.

`method` must support backtracking (`RungeKutta2` or `RungeKutta4`; `Euler` is not
supported). `grid_vxi` holds the staggered velocity grids and `grid` the chain's vertex
grid.
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

"""
    semilagrangian_advection!(chain::MarkerChain, method, V, grid_vxi, grid, dt)

Advance only the vertex topography `chain.h_vertices` by one semi-Lagrangian step.

Each vertex height is updated by backtracking its position through the velocity field `V`
(so `method` must support backtracking, i.e. `RungeKutta2`/`RungeKutta4`, not `Euler`). This
is the raw update used by [`semilagrangian_advection_markerchain!`](@ref); it does *not*
apply slope limiting, mass conservation, or marker reconstruction — call the wrapper unless
you need to compose those steps yourself.
"""
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

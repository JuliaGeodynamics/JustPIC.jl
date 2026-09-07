"""
    semilagrangian_advection_markerchain!(chain, method, V, grid_vxi, grid, dt; max_slope_angle = 45.0, conserve_mean = true)

Backtrack a marker chain through `V` and update the chain geometry with a
semi-Lagrangian step.

Unlike [`advect_markerchain!`](@ref), which moves the Lagrangian markers, this scheme
solves for the new vertex heights whose backward trajectories land on the old surface.
It then smooths slopes exceeding `max_slope_angle` degrees, restores the spatial mean
height, and rebuilds the markers. Smoothing is a single pass, not a strict slope bound;
use `max_slope_angle = nothing` (or 90 degrees) to disable it. Angles must be in `[0, 90]`.
Set `conserve_mean = false` when the
velocity field should change the mean height (for example, uniform uplift or boundary
flux). Mean conservation uses cell-width weights, including on refined grids.

`method` must support backtracking (`RungeKutta2` or `RungeKutta4`; `Euler` is not
supported). `grid_vxi` holds the staggered velocity grids and `grid` the chain's vertex
grid `(xv, yv)`, whose horizontal coordinates must match `chain.cell_vertices`.
The surface must remain single-valued; reduce `dt` if the characteristic solve fails.
"""
function semilagrangian_advection_markerchain!(
        chain::MarkerChain, method::AbstractAdvectionIntegrator, V, grid_vxi, grid, dt;
        max_slope_angle = 45.0, conserve_mean::Bool = true
    )
    angle = isnothing(max_slope_angle) ? 90 : max_slope_angle
    0 ≤ angle ≤ 90 || throw(ArgumentError("max_slope_angle must be in [0, 90] degrees or nothing"))
    target_mean = conserve_mean ? mean_height(chain) : nothing
    semilagrangian_advection!(chain, method, V, grid_vxi, grid, dt)
    angle < 90 && smooth_slopes!(chain, deg2rad(angle))
    return finish_markerchain_step!(chain, target_mean)
end

"""
    semilagrangian_advection!(chain::MarkerChain, method, V, grid_vxi, grid, dt)

Advance only the vertex topography `chain.h_vertices` by one semi-Lagrangian step.

Each new vertex height is found by backtracking through the velocity field `V`
(so `method` must support backtracking, i.e. `RungeKutta2`/`RungeKutta4`, not `Euler`). This
is the raw update used by [`semilagrangian_advection_markerchain!`](@ref); it does *not*
apply slope limiting, mass conservation, or marker reconstruction — call the wrapper unless
you need to compose those steps yourself. Departures outside the horizontal chain domain
sample the nearest endpoint height; velocity interpolation extrapolates from edge cells.
The old surface is piecewise linear, so RK order describes trajectory integration, not
the spatial interpolation order. A failed characteristic solve throws an error without
changing the chain.
"""
function semilagrangian_advection!(
        chain::MarkerChain,
        method::Union{RungeKutta2, RungeKutta4},
        V,
        grid_vxi::NTuple{N, NTuple{N, T}},
        grid,
        dt,
    ) where {N, T}
    (; h_vertices) = chain
    length(grid[1]) == length(h_vertices) ||
        throw(DimensionMismatch("The horizontal grid must match the chain vertices"))
    h_new = similar(h_vertices)

    # recast integrator/timestep/grids to the topography precision so Float32 backends
    # (e.g. Metal) are not silently promoted; `backend_grid` also makes the grids GPU-safe
    # when indexed directly inside the kernel (see advection!)
    Tc = eltype(h_vertices)
    backend = ka_backend(h_vertices)
    method = set_precision(method, Tc)
    dt = convert(Tc, dt)
    grid_vxi = backend_grid(backend, grid_vxi, Tc)
    xv = backend_grid(backend, grid[1], Tc)

    launch!(
        backend, semilagrangian_advection_markerchain_kernel!, length(h_vertices),
        h_new, h_vertices, method, V, grid_vxi, xv, dt
    )
    all(isfinite, h_new) || error("Marker-chain backtracking did not converge; reduce dt and check that the surface remains single-valued")
    copyto!(h_vertices, h_new)
    return nothing
end

# The trailing argument count is fixed at the marker-chain arity `(V, grid_vxi, grid, dt)`:
# an open `Vararg` would also cover the seven-argument grid-field
# `semilagrangian_advection!`, leaving the two methods ambiguous.
function semilagrangian_advection!(
        chain::MarkerChain,
        method::AbstractAdvectionIntegrator,
        args::Vararg{Any, 4},
    )
    throw(ArgumentError("Marker-chain backtracking requires RungeKutta2 or RungeKutta4"))
    return nothing
end

@kernel function semilagrangian_advection_markerchain_kernel!(
        h_new, h_old, method, V, grid_vxi, xv, dt
    )
    i = @index(Global)
    h_new[i] = backtrack_surface_height(h_old, method, V, grid_vxi, xv, dt, i)
end

@inline function backtrack_surface_height(h_old, method, V, grid_vxi, xv, dt, i)
    h = h_previous = h_old[i]
    residual_previous = zero(h)
    # Solve y_departure(h) - h_old(x_departure(h)) = 0 by secant iteration.
    # A single residual correction from the OLD height is only first-order accurate
    # for vertically varying velocity, regardless of the Runge-Kutta method.
    for iteration in 1:20
        xd, yd = advect_particle_markerchain(
            method, (xv[i], h), V, grid_vxi, nothing, dt, i; backtracking = true
        )
        isfinite(xd) && isfinite(yd) || break
        residual = yd - _interpolate_topography(xd, xv, h_old, i)
        abs(residual) ≤ 2 * eps(typeof(h)) * max(eps(typeof(h)), abs(h), abs(yd)) && return h
        denominator = residual - residual_previous
        correction = iteration == 1 || iszero(denominator) ? residual :
            residual * (h - h_previous) / denominator
        h_previous, residual_previous = h, residual
        h -= correction
        isfinite(h) || break
    end
    return oftype(h, NaN)
end

@inline function _interpolate_topography(xq, xv, h, seed)
    x = clamp(xq, xv[1], xv[end])
    i = clamp(parent_cell_index(x, xv, seed), 1, length(h) - 1)
    return _interp1D(x, xv[i], xv[i + 1], h[i], h[i + 1])
end

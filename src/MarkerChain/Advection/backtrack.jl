"""
    semilagrangian_advection_markerchain!(chain, method, V, grid_vxi, grid, dt; max_slope_angle = 45.0, conserve_mean = true)

Advect a marker chain through `V` and update the chain geometry with a
semi-Lagrangian step.

Unlike [`advect_markerchain!`](@ref), which moves the Lagrangian markers, this scheme
advects the old surface vertices and remaps them onto the fixed horizontal grid.
It then smooths slopes exceeding `max_slope_angle` degrees, restores the spatial mean
height, and rebuilds the markers. Smoothing is a single pass, not a strict slope bound;
use `max_slope_angle = nothing` (or 90 degrees) to disable it. Angles must be in `[0, 90]`.
Set `conserve_mean = false` when the
velocity field should change the mean height (for example, uniform uplift or boundary
flux). Mean conservation uses cell-width weights, including on refined grids.

`method` must be `RungeKutta2` or `RungeKutta4` (`Euler` is not supported). `grid_vxi` holds the staggered velocity grids and `grid` the chain's vertex
grid `(xv, yv)`, whose horizontal coordinates must match `chain.cell_vertices`.
The advected vertices must remain strictly ordered in x; reduce `dt` if they cross.
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

Each new vertex height is interpolated from forward-advected surface vertices through `V`
(`method` must be `RungeKutta2` or `RungeKutta4`, not `Euler`). This
is the raw update used by [`semilagrangian_advection_markerchain!`](@ref); it does *not*
apply slope limiting, mass conservation, or marker reconstruction — call the wrapper unless
you need to compose those steps yourself. Queries outside the advected horizontal chain domain
sample the nearest advected endpoint height; velocity interpolation extrapolates from edge cells.
The advected surface is piecewise linear, so RK order describes trajectory integration, not
the spatial interpolation order. Non-finite trajectories or crossed vertices throw an error without
changing the chain.
"""
function semilagrangian_advection!(
        chain::MarkerChain,
        method::Union{RungeKutta2, RungeKutta4},
        V,
        grid_vxi::NTuple{N, NTuple{N, Any}},
        grid,
        dt,
    ) where {N}
    (; h_vertices) = chain
    check_finite_scalar("dt", dt)
    check_velocity(V, grid_vxi, ka_backend(chain), eltype(h_vertices))
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
    all(isfinite, h_new) || error("Marker-chain remapping found non-finite trajectories or crossed vertices; reduce dt and check the velocity field")
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
    n = length(h_old)
    p = advect_particle_markerchain(method, (xv[i], h_old[i]), V, grid_vxi, nothing, dt, i)
    all(isfinite, p) || return oftype(h_old[i], NaN)
    if i < n
        q = advect_particle_markerchain(method, (xv[i + 1], h_old[i + 1]), V, grid_vxi, nothing, dt, i + 1)
        all(isfinite, q) && p[1] < q[1] || return oftype(h_old[i], NaN)
    end

    left = advect_particle_markerchain(method, (xv[1], h_old[1]), V, grid_vxi, nothing, dt, 1)
    right = advect_particle_markerchain(method, (xv[n], h_old[n]), V, grid_vxi, nothing, dt, n)
    x = xv[i]
    x ≤ left[1] && return left[2]
    x ≥ right[1] && return right[2]
    lo, hi = 1, n
    # Search a fixed set of advected vertices; no nonlinear characteristic solve.
    while hi - lo > 1
        mid = (lo + hi) ÷ 2
        p = advect_particle_markerchain(method, (xv[mid], h_old[mid]), V, grid_vxi, nothing, dt, mid)
        if p[1] ≤ x
            lo, left = mid, p
        else
            hi, right = mid, p
        end
    end
    return _interp1D(x, left[1], right[1], left[2], right[2])
end

"""
    semilagrangian_advection_markerchain!(chain, method, V, grid_vxi, grid, dt; max_slope_angle = 45.0, conserve_mean = true, periodic_1 = false)

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

With `periodic_1 = true` the x direction wraps: the two end vertices are one point of a surface
that repeats every `xv[end] - xv[1]`, so topography carried across one end re-enters at the
other, and the end vertices always hold the same height. Initialize those endpoints with equal
heights and supply `V` periodic in x, including its ghost values.

`method` must be `RungeKutta2` or `RungeKutta4` (`Euler` is not supported). `grid_vxi` holds the staggered velocity grids and `grid` the chain's vertex
grid `(xv, yv)`, whose horizontal coordinates must match `chain.cell_vertices`.
The advected vertices must remain strictly ordered in x; reduce `dt` if they cross.

Under an `ImplicitGlobalGrid` decomposition, `chain`, `V`, `grid_vxi` and `grid` are
those of the local subdomain, and `V` must have current halos. The advected vertices
are combined across each y-column and the mean height is that of the whole chain. A
vertex may move at most one cell along x per step, and must end inside its rank's
local grid along y unless it crosses the bottom or top of the global domain. The chain must lie inside the global domain, the x direction
must not be periodic, and every decomposed direction must have an overlap of 2; the
update raises on every rank otherwise. `periodic_1 = true` requires the x direction on a
single rank.
"""
function semilagrangian_advection_markerchain!(
        chain::MarkerChain, method::AbstractAdvectionIntegrator, V, grid_vxi, grid, dt;
        max_slope_angle = 45.0, conserve_mean::Bool = true, periodic_1::Bool = false
    )
    angle = isnothing(max_slope_angle) ? 90 : max_slope_angle
    0 ≤ angle ≤ 90 || throw(ArgumentError("max_slope_angle must be in [0, 90] degrees or nothing"))
    target_mean = conserve_mean ? _global_mean_height(chain.h_vertices, chain.cell_vertices) : nothing
    semilagrangian_advection!(chain, method, V, grid_vxi, grid, dt; periodic_1)
    angle < 90 && smooth_slopes!(chain, deg2rad(angle); periodic_1)
    return finish_markerchain_step!(chain, target_mean)
end

"""
    semilagrangian_advection!(chain::MarkerChain, method, V, grid_vxi, grid, dt; periodic_1 = false)

Advance only the vertex topography `chain.h_vertices` by one semi-Lagrangian step.

Each new vertex height is interpolated from forward-advected surface vertices through `V`
(`method` must be `RungeKutta2` or `RungeKutta4`, not `Euler`). This
is the raw update used by [`semilagrangian_advection_markerchain!`](@ref); it does *not*
apply slope limiting, mass conservation, or marker reconstruction — call the wrapper unless
you need to compose those steps yourself. Queries outside the advected horizontal chain domain
sample the nearest advected endpoint height; velocity interpolation extrapolates from edge cells.
The advected surface is piecewise linear, so RK order describes trajectory integration, not
the spatial interpolation order. Non-finite trajectories or crossed vertices throw an error without
changing the chain. With `periodic_1 = true` the intermediate stages of a trajectory wrap around
the x direction and the remap closes the polyline across the seam, so no query falls outside it.
"""
function semilagrangian_advection!(
        chain::MarkerChain,
        method::Union{RungeKutta2, RungeKutta4},
        V,
        grid_vxi::NTuple{N, NTuple{N, Any}},
        grid,
        dt;
        periodic_1::Bool = false,
    ) where {N}
    (; h_vertices) = chain
    _check_chain_grid()
    _check_periodic_chain(periodic_1)
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

    px, py = similar(h_vertices), similar(h_vertices)
    _advect_chain_vertices!(px, py, h_vertices, method, V, grid_vxi, xv, dt, periodic_1)
    _reduce_chain_positions_y!(px, py, h_vertices, grid_vxi[2][2])
    _remap_chain_heights!(h_new, px, py, xv, periodic_1)
    _check_chain_remap(h_new, px, xv)
    copyto!(h_vertices, h_new)
    _update_chain_halo!(h_vertices)
    return nothing
end

# The trailing argument count is fixed at the marker-chain arity `(V, grid_vxi, grid, dt)`:
# an open `Vararg` would also cover the seven-argument grid-field
# `semilagrangian_advection!`, leaving the two methods ambiguous.
function semilagrangian_advection!(
        chain::MarkerChain,
        method::AbstractAdvectionIntegrator,
        args::Vararg{Any, 4};
        kwargs...,
    )
    throw(ArgumentError("Marker-chain backtracking requires RungeKutta2 or RungeKutta4"))
    return nothing
end

"""
    _advect_chain_vertices!(px, py, h, method, V, grid_vxi, xv, dt, periodic)

Store in `(px[i], py[i])` the position of the chain vertex `(xv[i], h[i])` advected
through `V` over `dt`. With `periodic`, the intermediate stages wrap into `[xv[1], xv[end])`
and the final position is left unwrapped, so the advected polyline stays ordered.
"""
function _advect_chain_vertices!(px, py, h, method, V, grid_vxi, xv, dt, periodic)
    launch!(
        ka_backend(h), advect_chain_vertices_kernel!, length(h),
        px, py, h, method, V, grid_vxi, xv, dt, periodic
    )
    return nothing
end

@kernel function advect_chain_vertices_kernel!(px, py, h, method, V, grid_vxi, xv, dt, periodic)
    i = @index(Global)
    xlims = (xv[1], xv[length(xv)])
    p = advect_particle_markerchain(method, (xv[i], h[i]), V, grid_vxi, nothing, dt, i; periodic, xlims)
    px[i], py[i] = p
end

"""
    _remap_chain_heights!(h_new, px, py, xv, periodic)

Sample the polyline through the advected vertices `(px, py)` at the vertices `xv`.
Queries outside the advected polyline take the height of its nearest end, or, with
`periodic`, wrap around the seam. `h_new[i]` is `NaN` where the trajectory of vertex `i` is
non-finite or where vertices `i` and `i + 1` crossed.
"""
function _remap_chain_heights!(h_new, px, py, xv, periodic)
    launch!(ka_backend(h_new), remap_chain_heights_kernel!, length(h_new), h_new, px, py, xv, periodic)
    return nothing
end

@kernel function remap_chain_heights_kernel!(h_new, px, py, xv, periodic)
    i = @index(Global)
    h_new[i] = periodic ? _remap_chain_height_periodic(px, py, xv, i) : _remap_chain_height(px, py, xv, i)
end

@inline function _remap_chain_height(px, py, xv, i)
    n = length(px)
    nan = oftype(py[i], NaN)
    isfinite(px[i]) && isfinite(py[i]) || return nan
    if i < n
        isfinite(px[i + 1]) && isfinite(py[i + 1]) && px[i] < px[i + 1] || return nan
    end
    x = xv[i]
    x ≤ px[1] && return py[1]
    x ≥ px[n] && return py[n]
    lo, hi = 1, n
    # Search a fixed set of advected vertices; no nonlinear characteristic solve.
    while hi - lo > 1
        mid = (lo + hi) ÷ 2
        if px[mid] ≤ x
            lo = mid
        else
            hi = mid
        end
    end
    return _interp1D(x, px[lo], px[hi], py[lo], py[hi])
end

# The last vertex is the first one a period further on: it closes the polyline across the seam
# as `(px[1] + L, py[1])`, and every query is folded into `[px[1], px[1] + L)`. The last vertex
# takes the query of the first, so both ends get the same height.
@inline function _remap_chain_height_periodic(px, py, xv, i)
    n = length(px)
    nan = oftype(py[i], NaN)
    isfinite(px[i]) && isfinite(py[i]) || return nan
    if i < n
        isfinite(px[i + 1]) && isfinite(py[i + 1]) && px[i] < px[i + 1] || return nan
    end
    L = xv[n] - xv[1]
    x0 = px[1]
    x = x0 + mod(xv[ifelse(i == n, 1, i)] - x0, L)
    lo, hi = 1, n
    while hi - lo > 1
        mid = (lo + hi) ÷ 2
        if px[mid] ≤ x
            lo = mid
        else
            hi = mid
        end
    end
    x_hi, y_hi = hi == n ? (x0 + L, py[1]) : (px[hi], py[hi])
    return _interp1D(x, px[lo], x_hi, py[lo], y_hi)
end

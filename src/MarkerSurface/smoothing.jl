"""
    smooth_surface_max_angle!(surf::MarkerSurface, max_slope_angle)

Smooth the topography where the slope angle exceeds `max_slope_angle` (in **degrees**,
matching the MarkerChain `smooth_slopes!` convention).

This mirrors LaMEM's `FreeSurfSmoothMaxAngle`:
1. Scan all cells, compute the max slope (tan) from the 4 corner nodes and
   the average cell height; mark cells exceeding `tan(max_angle)`.
2. For each node touching at least one marked cell, replace its topography
   with the average of the (up to 4) surrounding cell-center heights.

# Arguments
- `surf`            : the `MarkerSurface`
- `max_slope_angle` : maximum slope angle in **degrees** (e.g. `45.0`)
"""
function smooth_surface_max_angle!(surf::MarkerSurface, max_slope_angle::Real)
    max_slope_angle ≤ 0 && return nothing

    topo = surf.topo
    xv = surf.xv
    yv = surf.yv
    nx1, ny1 = size(topo)
    nx = nx1 - 1  # number of cells in x
    ny = ny1 - 1  # number of cells in y

    tan_max = tan(deg2rad(max_slope_angle))

    # Step 1: cell-average heights + mask of cells exceeding max slope.
    # A separate Bool mask (not a sign-encoded height) so topographies ≤ 0 work.
    cell_topo = similar(topo, nx, ny)
    steep = similar(topo, Bool, nx, ny)

    launch!(
        ka_backend(topo), _smooth_step1_kernel!, (nx, ny),
        cell_topo, steep, topo, xv, yv, tan_max
    )

    # Step 2: smooth nodal topography (no-op per node when no neighbours are marked)
    launch!(
        ka_backend(topo), _smooth_step2_kernel!, (nx1, ny1),
        topo, cell_topo, steep, nx, ny, surf.periodic_1, surf.periodic_2
    )

    # MPI: sync boundary nodes smoothed with clamped (one-sided) cell stencils
    update_surface_halo!(surf)

    return nothing
end

@kernel function _smooth_step1_kernel!(
        cell_topo, steep, topo, xv, yv, tan_max
    )
    ic, jc = @index(Global, NTuple)
    # Cell sizes
    dx = xv[ic + 1] - xv[ic]
    dy = yv[jc + 1] - yv[jc]

    # Corner topography
    z00 = topo[ic, jc]
    z10 = topo[ic + 1, jc]
    z01 = topo[ic, jc + 1]
    z11 = topo[ic + 1, jc + 1]

    # Maximum slope (tangent)
    tmax = abs(z10 - z00) / dx
    t = abs(z11 - z01) / dx; tmax = max(tmax, t)
    t = abs(z01 - z00) / dy; tmax = max(tmax, t)
    t = abs(z11 - z10) / dy; tmax = max(tmax, t)

    # Average cell height; steepness recorded in a separate mask
    cell_topo[ic, jc] = (z00 + z10 + z01 + z11) / 4
    steep[ic, jc] = tmax > tan_max
end

@kernel function _smooth_step2_kernel!(
        topo, cell_topo, steep, nx, ny, periodic_1, periodic_2
    )
    iv, jv = @index(Global, NTuple)
    # Get the up-to-4 neighboring cell indices
    # When periodic: wrap with mod1 (LaMEM style: only clamp when NOT periodic)
    i1 = periodic_1 ? mod1(iv, nx) : min(iv, nx)
    i2 = periodic_1 ? mod1(iv - 1, nx) : max(iv - 1, 1)
    j1 = periodic_2 ? mod1(jv, ny) : min(jv, ny)
    j2 = periodic_2 ? mod1(jv - 1, ny) : max(jv - 1, 1)

    if steep[i1, j1] || steep[i1, j2] || steep[i2, j1] || steep[i2, j2]
        topo[iv, jv] =
            (cell_topo[i1, j1] + cell_topo[i1, j2] + cell_topo[i2, j1] + cell_topo[i2, j2]) / 4
    end
end

"""
    smooth_surface_diffusive!(surf::MarkerSurface, niter::Int=1; weight=0.25)

Apply diffusive smoothing to the topography. Each iteration replaces each interior
node with a weighted average of its neighbors.

# Arguments
- `surf`   : the `MarkerSurface`
- `niter`  : number of smoothing iterations (default `1`)
- `weight` : diffusion weight ∈ (0, 1) (default `0.25`)
"""
function smooth_surface_diffusive!(surf::MarkerSurface, niter::Int = 1; weight::Float64 = 0.25)
    topo = surf.topo
    nx1, ny1 = size(topo)
    buf = similar(topo)

    for _ in 1:niter
        copyto!(buf, topo)
        launch!(
            ka_backend(topo), _smooth_diffusive_kernel!, (nx1, ny1),
            topo, buf, weight, nx1, ny1
        )
        # MPI: boundary nodes are interior on the neighbouring rank
        update_surface_halo!(surf)
    end

    return nothing
end

# Boundary nodes (i or j at the domain edge) are left untouched — they stay at
# their pre-iteration `buf` value, synced from the neighbouring rank by the
# halo exchange that follows each iteration.
@kernel function _smooth_diffusive_kernel!(
        topo, buf, weight, nx1, ny1
    )
    i, j = @index(Global, NTuple)
    if 1 < i < nx1 && 1 < j < ny1
        avg = (buf[i - 1, j] + buf[i + 1, j] + buf[i, j - 1] + buf[i, j + 1]) / 4
        topo[i, j] = (1 - weight) * buf[i, j] + weight * avg
    end
end

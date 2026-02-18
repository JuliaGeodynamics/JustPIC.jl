"""
    smooth_surface_max_angle!(surf::MarkerSurface, max_slope_angle)

Smooth the topography where the slope angle exceeds `max_slope_angle` (in **degrees**,
matching the MarkerChain `smooth_slopes!` convention).

This mirrors LaMEM's `FreeSurfSmoothMaxAngle`:
1. Scan all cells, compute the max slope (tan) from the 4 corner nodes.
2. If any cell exceeds `tan(max_angle)`, mark it and store the average cell height
   with a negative sign.
3. For each affected node, replace its topography with the average of surrounding
   cell-center heights (only cells that were marked).

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

    # Step 1: compute cell-center topography and mark cells exceeding max slope
    # Use pre-allocated workspace buffers
    cell_topo = surf.workspace.cell_topo
    affected = surf.workspace.affected
    fill!(cell_topo, 0.0)
    fill!(affected, Int32(0))

    @parallel (1:nx, 1:ny) _smooth_step1_kernel!(
        cell_topo, affected, topo, xv, yv, tan_max
    )

    # If no cells are affected, return
    any(!iszero, affected) || return nothing

    # Step 2: smooth nodal topography
    @parallel (1:nx1, 1:ny1) _smooth_step2_kernel!(
        topo, cell_topo, affected, nx, ny
    )

    return nothing
end

@parallel_indices (ic, jc) function _smooth_step1_kernel!(
        cell_topo, affected, topo, xv, yv, tan_max
    )
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

    # Average cell height
    h = (z00 + z10 + z01 + z11) / 4

    if tmax > tan_max
        cell_topo[ic, jc] = -h   # mark with negative sign
        affected[ic, jc] = Int32(1)
    else
        cell_topo[ic, jc] = h
        affected[ic, jc] = Int32(0)
    end

    return nothing
end

@parallel_indices (iv, jv) function _smooth_step2_kernel!(
        topo, cell_topo, affected, nx, ny
    )
    # Get the up-to-4 neighboring cell indices
    i1 = min(iv, nx)  # right cell
    i2 = max(iv - 1, 1)   # left cell
    j1 = min(jv, ny)  # top cell
    j2 = max(jv - 1, 1)   # bottom cell

    cz1 = cell_topo[i1, j1]
    cz2 = cell_topo[i1, j2]
    cz3 = cell_topo[i2, j1]
    cz4 = cell_topo[i2, j2]

    # Check if any neighbor is affected and get absolute values
    cnt = 0
    czabs1 = cz1
    czabs2 = cz2
    czabs3 = cz3
    czabs4 = cz4

    if cz1 < 0
        cnt += 1
        czabs1 = -cz1
    end
    if cz2 < 0
        cnt += 1
        czabs2 = -cz2
    end
    if cz3 < 0
        cnt += 1
        czabs3 = -cz3
    end
    if cz4 < 0
        cnt += 1
        czabs4 = -cz4
    end

    if cnt > 0
        topo[iv, jv] = (czabs1 + czabs2 + czabs3 + czabs4) / 4
    end

    return nothing
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
    buf = surf.workspace.buf

    for _ in 1:niter
        copyto!(buf, topo)
        @parallel (2:(nx1 - 1), 2:(ny1 - 1)) _smooth_diffusive_kernel!(
            topo, buf, weight
        )
    end

    return nothing
end

@parallel_indices (i, j) function _smooth_diffusive_kernel!(
        topo, buf, weight
    )
    avg = (buf[i - 1, j] + buf[i + 1, j] + buf[i, j - 1] + buf[i, j + 1]) / 4
    topo[i, j] = (1 - weight) * buf[i, j] + weight * avg

    return nothing
end

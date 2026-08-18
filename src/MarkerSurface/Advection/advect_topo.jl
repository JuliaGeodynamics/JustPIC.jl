"""
    _ghost_coord(v, i, n)

Return the `i`-th coordinate of vector `v` (length `n`) allowing one ghost index
on each side (`i == 0` and `i == n+1`) via linear extrapolation. Used to build
the deformed-grid stencil at domain boundaries without materialising a padded
array.
"""
@inline function _ghost_coord(v, i, n)
    return if i == 0
        2 * v[1] - v[2]
    elseif i == n + 1
        2 * v[n] - v[n - 1]
    else
        v[i]
    end
end

"""
    _ghost_field(arr, i, j, nx, ny, periodic_1, periodic_2)

Return the value of 2D field `arr` (size `(nx, ny)`) at index `(i, j)`, allowing
one ghost layer on each side (`i ∈ 0:nx+1`, `j ∈ 0:ny+1`). Ghost values wrap to
the opposite boundary when periodic; otherwise they clamp to the boundary
node. Coordinates use linear extrapolation separately in `_ghost_coord`.
"""
@inline function _ghost_field(arr, i, j, nx, ny, periodic_1::Bool, periodic_2::Bool)
    return @inbounds begin
        ii = i == 0 ? (periodic_1 ? nx - 1 : 1) :
            i == nx + 1 ? (periodic_1 ? 2 : nx) : i
        jj = j == 0 ? (periodic_2 ? ny - 1 : 1) :
            j == ny + 1 ? (periodic_2 ? 2 : ny) : j
        return arr[ii, jj]
    end
end

"""
     advect_surface_topo!(surf::MarkerSurface, dt)

Advect the topography on the free surface mesh using the velocity field
already interpolated onto the surface nodes (`surf.vx`, `surf.vy`, `surf.vz`).

1. Extrapolate ghost coordinates while clamping nonperiodic field ghosts to
    boundary nodes, avoiding degenerate stencils at domain boundaries.
2. For each surface node, build a local 3×3 "deformed grid" using neighboring
    node positions displaced by `dt*v`.
3. Subdivide the deformed cell into 16 triangles (9 corner + 4 midpoint nodes).
4. Find which triangle contains the target position and perform barycentric interpolation of the z-coordinate.
```text
    1 ------- 2 ------- 3
    |  \\     / \\     /  |
    |   \\   /   \\   /   |
    |    \\ /     \\ /    |
    |    10       11    |
    |    / \\     / \\    |
    |   /   \\   /   \\   |
    |  /     \\ /     \\  |
    4 ------- 5 ------- 6
    |  \\     / \\     /  |
    |   \\   /   \\   /   |
    |    \\ /     \\ /    |
    |    12       13    |
    |    / \\     / \\    |
    |   /   \\   /   \\   |
    |  /     \\ /     \\  |
    7 ------- 8 ------- 9
```
# Arguments
- `surf` : the `MarkerSurface`
- `dt`   : time step
"""
function advect_surface_topo!(surf::MarkerSurface, dt)
    xv = surf.xv
    yv = surf.yv
    topo = surf.topo
    topo0 = surf.topo0
    vx = surf.vx
    vy = surf.vy
    vz = surf.vz
    nx1, ny1 = size(topo)

    periodic_1 = surf.periodic_1
    periodic_2 = surf.periodic_2
    dt = convert(eltype(topo), dt)

    _surface_collective_failure(!all(isfinite, topo)) &&
        throw(ArgumentError("MarkerSurface topography must be finite before advection"))

    # Save old topography; the advection stencil reads from `topo0` and writes into
    # `topo`, so no separate scratch buffer is needed. Ghost nodes are handled
    # in-kernel by `_ghost_coord`/`_ghost_field`, avoiding any padded-array allocation.
    copyto!(topo0, topo)

    valid = surf.advection_valid
    launch!(
        ka_backend(topo), _advect_surface_topo_kernel!, (nx1, ny1),
        topo, valid, topo0, xv, yv, vx, vy, vz, dt, nx1, ny1, periodic_1, periodic_2
    )

    if _surface_collective_failure(!all(valid))
        # Nodes that did find a containing triangle have already been written, so
        # roll the whole surface back: a caller retrying with a smaller time step
        # must start from the same topography on every rank.
        copyto!(topo, topo0)
        throw(ArgumentError("MarkerSurface advection target is outside its deformed-grid stencil; reduce the time step or inspect the velocity field"))
    end

    # MPI: replace ghost-extrapolated boundary nodes with neighbour values
    update_surface_halo!(surf)

    # Keep the two redundant boundary nodes in sync (periodic seam)
    _enforce_periodic_seam!(topo, periodic_1, periodic_2)

    return nothing
end

@inline function _surface_collective_failure(local_failure)
    ImplicitGlobalGrid.grid_is_initialized() || return local_failure
    return MPI.Allreduce(local_failure, |, ImplicitGlobalGrid.global_grid().comm)
end

function _enforce_periodic_seam!(topo::AbstractMatrix, periodic_1::Bool, periodic_2::Bool)
    if periodic_1
        @views topo[end, :] .= topo[1, :]
    end
    if periodic_2
        @views topo[:, end] .= topo[:, 1]
    end
    return nothing
end

@kernel function _advect_surface_topo_kernel!(
        topo, valid, topo0, xv, yv, vx, vy, vz, dt, nx1, ny1, periodic_1, periodic_2
    )
    i, j = @index(Global, NTuple)
    # The 16-triangle subdivision topology
    # Vertices 1-9 are the 3x3 grid nodes, 10-13 are cell-center midpoints
    tria = (
        # Inner layer (center-connected triangles)
        (5, 6, 13), (5, 13, 8), (5, 8, 12), (5, 12, 4),
        (5, 4, 10), (5, 10, 2), (5, 2, 11), (5, 11, 6),
        # Outer layer (corner triangles)
        (6, 9, 13), (9, 8, 13), (8, 7, 12), (7, 4, 12),
        (4, 1, 10), (1, 2, 10), (2, 3, 11), (3, 6, 11),
    )

    # Node coordinates
    X = xv[i]
    Y = yv[j]

    # Neighbor coordinates (ghost-aware, always non-degenerate)
    X1 = _ghost_coord(xv, i - 1, nx1)
    X2 = _ghost_coord(xv, i + 1, nx1)
    Y1 = _ghost_coord(yv, j - 1, ny1)
    Y2 = _ghost_coord(yv, j + 1, ny1)

    # Build 9 deformed grid positions (node + dt*velocity), reading the old
    # topography/velocity through ghost-aware accessors instead of padded arrays.
    # Ordering: (j-row, i-col) → linear index
    #  1=(j-1,i-1) 2=(j-1,i) 3=(j-1,i+1)
    #  4=(j  ,i-1) 5=(j  ,i) 6=(j  ,i+1)
    #  7=(j+1,i-1) 8=(j+1,i) 9=(j+1,i+1)
    cx = (
        dt * _ghost_field(vx, i - 1, j - 1, nx1, ny1, periodic_1, periodic_2) + X1,  # 1
        dt * _ghost_field(vx, i, j - 1, nx1, ny1, periodic_1, periodic_2) + X,   # 2
        dt * _ghost_field(vx, i + 1, j - 1, nx1, ny1, periodic_1, periodic_2) + X2,  # 3
        dt * _ghost_field(vx, i - 1, j, nx1, ny1, periodic_1, periodic_2) + X1,  # 4
        dt * _ghost_field(vx, i, j, nx1, ny1, periodic_1, periodic_2) + X,   # 5
        dt * _ghost_field(vx, i + 1, j, nx1, ny1, periodic_1, periodic_2) + X2,  # 6
        dt * _ghost_field(vx, i - 1, j + 1, nx1, ny1, periodic_1, periodic_2) + X1,  # 7
        dt * _ghost_field(vx, i, j + 1, nx1, ny1, periodic_1, periodic_2) + X,   # 8
        dt * _ghost_field(vx, i + 1, j + 1, nx1, ny1, periodic_1, periodic_2) + X2,  # 9
    )

    cy = (
        dt * _ghost_field(vy, i - 1, j - 1, nx1, ny1, periodic_1, periodic_2) + Y1,  # 1
        dt * _ghost_field(vy, i, j - 1, nx1, ny1, periodic_1, periodic_2) + Y1,  # 2
        dt * _ghost_field(vy, i + 1, j - 1, nx1, ny1, periodic_1, periodic_2) + Y1,  # 3
        dt * _ghost_field(vy, i - 1, j, nx1, ny1, periodic_1, periodic_2) + Y,   # 4
        dt * _ghost_field(vy, i, j, nx1, ny1, periodic_1, periodic_2) + Y,   # 5
        dt * _ghost_field(vy, i + 1, j, nx1, ny1, periodic_1, periodic_2) + Y,   # 6
        dt * _ghost_field(vy, i - 1, j + 1, nx1, ny1, periodic_1, periodic_2) + Y2,  # 7
        dt * _ghost_field(vy, i, j + 1, nx1, ny1, periodic_1, periodic_2) + Y2,  # 8
        dt * _ghost_field(vy, i + 1, j + 1, nx1, ny1, periodic_1, periodic_2) + Y2,  # 9
    )

    cz = (
        dt * _ghost_field(vz, i - 1, j - 1, nx1, ny1, periodic_1, periodic_2) + _ghost_field(topo0, i - 1, j - 1, nx1, ny1, periodic_1, periodic_2),  # 1
        dt * _ghost_field(vz, i, j - 1, nx1, ny1, periodic_1, periodic_2) + _ghost_field(topo0, i, j - 1, nx1, ny1, periodic_1, periodic_2),  # 2
        dt * _ghost_field(vz, i + 1, j - 1, nx1, ny1, periodic_1, periodic_2) + _ghost_field(topo0, i + 1, j - 1, nx1, ny1, periodic_1, periodic_2),  # 3
        dt * _ghost_field(vz, i - 1, j, nx1, ny1, periodic_1, periodic_2) + _ghost_field(topo0, i - 1, j, nx1, ny1, periodic_1, periodic_2),  # 4
        dt * _ghost_field(vz, i, j, nx1, ny1, periodic_1, periodic_2) + _ghost_field(topo0, i, j, nx1, ny1, periodic_1, periodic_2),  # 5
        dt * _ghost_field(vz, i + 1, j, nx1, ny1, periodic_1, periodic_2) + _ghost_field(topo0, i + 1, j, nx1, ny1, periodic_1, periodic_2),  # 6
        dt * _ghost_field(vz, i - 1, j + 1, nx1, ny1, periodic_1, periodic_2) + _ghost_field(topo0, i - 1, j + 1, nx1, ny1, periodic_1, periodic_2),  # 7
        dt * _ghost_field(vz, i, j + 1, nx1, ny1, periodic_1, periodic_2) + _ghost_field(topo0, i, j + 1, nx1, ny1, periodic_1, periodic_2),  # 8
        dt * _ghost_field(vz, i + 1, j + 1, nx1, ny1, periodic_1, periodic_2) + _ghost_field(topo0, i + 1, j + 1, nx1, ny1, periodic_1, periodic_2),  # 9
    )

    # 4 midpoints (averages of cell-center quartets)
    cx10 = (cx[1] + cx[2] + cx[4] + cx[5]) / 4
    cy10 = (cy[1] + cy[2] + cy[4] + cy[5]) / 4
    cz10 = (cz[1] + cz[2] + cz[4] + cz[5]) / 4

    cx11 = (cx[2] + cx[3] + cx[5] + cx[6]) / 4
    cy11 = (cy[2] + cy[3] + cy[5] + cy[6]) / 4
    cz11 = (cz[2] + cz[3] + cz[5] + cz[6]) / 4

    cx12 = (cx[4] + cx[5] + cx[7] + cx[8]) / 4
    cy12 = (cy[4] + cy[5] + cy[7] + cy[8]) / 4
    cz12 = (cz[4] + cz[5] + cz[7] + cz[8]) / 4

    cx13 = (cx[5] + cx[6] + cx[8] + cx[9]) / 4
    cy13 = (cy[5] + cy[6] + cy[8] + cy[9]) / 4
    cz13 = (cz[5] + cz[6] + cz[8] + cz[9]) / 4

    # Extended coordinate arrays (13 points)
    all_cx = (cx..., cx10, cx11, cx12, cx13)
    all_cy = (cy..., cy10, cy11, cy12, cy13)
    all_cz = (cz..., cz10, cz11, cz12, cz13)

    # The target is the fixed Eulerian surface vertex. Any background strain must
    # be represented in the supplied velocity field.
    Xt = X
    Yt = Y

    # Search through the 16 triangles
    Z = zero(eltype(topo))
    found = false

    for tri in tria
        ok, Zinterp = _interpolate_triangle(
            all_cx, all_cy, all_cz, tri, Xt, Yt
        )
        if ok
            Z = Zinterp
            found = true
            break
        end
    end

    topo[i, j] = ifelse(found, Z, topo0[i, j])
    valid[i, j] = found
end

"""
    _interpolate_triangle(cx, cy, cz, tri, xp, yp; tol=1e-10)

Check if point (xp, yp) lies inside the triangle defined by indices `tri`
into coordinate arrays `(cx, cy, cz)`, and compute the barycentric
interpolation of the z-coordinate.

# Returns
- `(true, z_interpolated)` if the point is inside the triangle
- `(false, zero(T))` otherwise
"""
@inline function _interpolate_triangle(
        cx::NTuple, cy::NTuple, cz::NTuple,
        tri::NTuple{3, Int}, xp::T, yp::T;
        tol = convert(T, 1.0e-6),
    ) where {T}
    ia, ib, ic = tri

    xa, ya = cx[ia], cy[ia]
    xb, yb = cx[ib], cy[ib]
    xc, yc = cx[ic], cy[ic]

    # Sub-triangle areas (× 2)
    la = abs((xp - xc) * (yb - yc) - (xb - xc) * (yp - yc))
    lb = abs((xp - xa) * (yc - ya) - (xc - xa) * (yp - ya))
    lc = abs((xp - xb) * (ya - yb) - (xa - xb) * (yp - yb))

    # Total triangle area (× 2)
    A = abs((xa - xc) * (yb - yc) - (xb - xc) * (ya - yc))

    S = la + lb + lc

    # Point test
    if S > A * (one(T) + tol)
        return false, zero(T)
    end

    # Normalize barycentric coordinates
    if S > zero(T)
        la /= S
        lb /= S
        lc /= S
    else
        la = lb = lc = one(T) / 3
    end

    # Interpolate z
    fp = la * cz[ia] + lb * cz[ib] + lc * cz[ic]

    return true, fp
end

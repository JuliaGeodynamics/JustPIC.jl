"""
    _pad_extrap_1d(arr::AbstractVector)

Pad a 1D array with one ghost element on each side using linear extrapolation.
Returns a new array of length `length(arr) + 2`.
"""
function _pad_extrap_1d(arr::AbstractVector)
    n = length(arr)
    p = similar(arr, n + 2)
    _pad_extrap_1d!(p, arr)
    return p
end

"""
    _pad_extrap_1d!(p, arr)

In-place version: fill pre-allocated `p` (length `n+2`) from `arr` (length `n`).
"""
function _pad_extrap_1d!(p::AbstractVector, arr::AbstractVector)
    n = length(arr)
    # Interior copy
    @parallel (1:n) _pad_copy_1d_kernel!(p, arr)
    # Boundary extrapolation (single thread, no branching)
    @parallel (1:1) _pad_boundary_1d_kernel!(p, arr, n)
    return nothing
end

@parallel_indices (i) function _pad_copy_1d_kernel!(p, arr)
    @inbounds p[i + 1] = arr[i]
    return nothing
end

@parallel_indices (i) function _pad_boundary_1d_kernel!(p, arr, n)
    @inbounds p[1]     = 2 * arr[1] - arr[2]
    @inbounds p[n + 2] = 2 * arr[n] - arr[n - 1]
    return nothing
end

"""
    _pad_extrap_2d(arr::AbstractMatrix)

Pad a 2D array with one ghost layer on each side using linear extrapolation.
Returns a new array of size `(size(arr,1)+2, size(arr,2)+2)`.
"""
function _pad_extrap_2d(arr::AbstractMatrix)
    nx, ny = size(arr)
    p = similar(arr, nx + 2, ny + 2)
    _pad_extrap_2d!(p, arr)
    return p
end

"""
    _pad_extrap_2d!(p, arr)

In-place version: fill pre-allocated `p` (size `(nx+2, ny+2)`) from `arr` (size `(nx, ny)`).
"""
function _pad_extrap_2d!(p::AbstractMatrix, arr::AbstractMatrix)
    nx, ny = size(arr)
    # Interior copy
    @parallel (1:nx, 1:ny) _pad_copy_2d_kernel!(p, arr)
    # Edge extrapolation
    @parallel (1:ny) _pad_left_edge_kernel!(p, arr)
    @parallel (1:ny) _pad_right_edge_kernel!(p, arr, nx)
    @parallel (1:nx) _pad_bottom_edge_kernel!(p, arr)
    @parallel (1:nx) _pad_top_edge_kernel!(p, arr, ny)
    # Corner extrapolation (single thread, no branching)
    @parallel (1:1) _pad_corners_2d_kernel!(p, arr, nx, ny)
    return nothing
end

@parallel_indices (i, j) function _pad_copy_2d_kernel!(p, arr)
    @inbounds p[i + 1, j + 1] = arr[i, j]
    return nothing
end

@parallel_indices (j) function _pad_left_edge_kernel!(p, arr)
    @inbounds p[1, j + 1] = 2 * arr[1, j] - arr[2, j]
    return nothing
end

@parallel_indices (j) function _pad_right_edge_kernel!(p, arr, nx)
    @inbounds p[nx + 2, j + 1] = 2 * arr[nx, j] - arr[nx - 1, j]
    return nothing
end

@parallel_indices (i) function _pad_bottom_edge_kernel!(p, arr)
    @inbounds p[i + 1, 1] = 2 * arr[i, 1] - arr[i, 2]
    return nothing
end

@parallel_indices (i) function _pad_top_edge_kernel!(p, arr, ny)
    @inbounds p[i + 1, ny + 2] = 2 * arr[i, ny] - arr[i, ny - 1]
    return nothing
end

@parallel_indices (i) function _pad_corners_2d_kernel!(p, arr, nx, ny)
    # Bottom-left
    @inbounds p[1, 1] = 4 * arr[1, 1] - 2 * arr[1, 2] - 2 * arr[2, 1] + arr[2, 2]
    # Top-left
    @inbounds p[1, ny + 2] = 4 * arr[1, ny] - 2 * arr[1, ny - 1] - 2 * arr[2, ny] + arr[2, ny - 1]
    # Bottom-right
    @inbounds p[nx + 2, 1] = 4 * arr[nx, 1] - 2 * arr[nx, 2] - 2 * arr[nx - 1, 1] + arr[nx - 1, 2]
    # Top-right
    @inbounds p[nx + 2, ny + 2] = 4 * arr[nx, ny] - 2 * arr[nx, ny - 1] - 2 * arr[nx - 1, ny] + arr[nx - 1, ny - 1]
    return nothing
end

"""
     advect_surface_topo!(surf::MarkerSurface, dt; Exx=0.0, Eyy=0.0)

Advect the topography on the free surface mesh using the velocity field
already interpolated onto the surface nodes (`surf.vx`, `surf.vy`, `surf.vz`).

1. Pad topography and velocity arrays with linearly extrapolated ghost nodes
    to avoid degenerate stencils at domain boundaries.
2. For each surface node, build a local 3×3 "deformed grid" using neighboring
    node positions displaced by `dt*v`.
3. Subdivide the deformed cell into 16 triangles (9 corner + 4 midpoint nodes).
4. Find which triangle contains the target position and perform barycentric interpolation of the z-coordinate.

# Arguments
- `surf` : the `MarkerSurface`
- `dt`   : time step
- `Exx, Eyy` : background strain rates in x,y directions (default `0.0`)
"""
function advect_surface_topo!(
     surf::MarkerSurface, dt;
     Exx = 0.0, Eyy = 0.0)
    xv = surf.xv
    yv = surf.yv
    topo = surf.topo
    vx = surf.vx
    vy = surf.vy
    vz = surf.vz
    nx1, ny1 = size(topo)

    # Save old topography
    copyto!(surf.topo0, topo)

    # Use pre-allocated workspace buffers
    ws = surf.workspace
    advected = ws.advected
    xvp = ws.xvp
    yvp = ws.yvp
    topop = ws.topop
    vxp = ws.vxp
    vyp = ws.vyp
    vzp = ws.vzp

    # Fill padded arrays in-place
    _pad_extrap_1d!(xvp, xv)
    _pad_extrap_1d!(yvp, yv)
    _pad_extrap_2d!(topop, topo)
    _pad_extrap_2d!(vxp, vx)
    _pad_extrap_2d!(vyp, vy)
    _pad_extrap_2d!(vzp, vz)

    @parallel (1:nx1, 1:ny1) _advect_surface_topo_kernel!(
        advected, xvp, yvp, topop, vxp, vyp, vzp, dt, Exx, Eyy
    )

    # Copy advected topography back
    copyto!(topo, advected)

    return nothing
end

@parallel_indices (i, j) function _advect_surface_topo_kernel!(
        advected, xvp, yvp, topop, vxp, vyp, vzp, dt, Exx, Eyy
    )
    # The 16-triangle subdivision topology (0-indexed vertex IDs → 1-indexed)
    # Vertices 0-8 are the 3x3 grid nodes, 9-12 are cell-center midpoints
    tria = (
        # Inner layer (center-connected triangles)
        (5, 6, 13), (5, 13, 8), (5, 8, 12), (5, 12, 4),
        (5, 4, 10), (5, 10, 2), (5, 2, 11), (5, 11, 6),
        # Outer layer (corner triangles)
        (6, 9, 13), (9, 8, 13), (8, 7, 12), (7, 4, 12),
        (4, 1, 10), (1, 2, 10), (2, 3, 11), (3, 6, 11),
    )
    # Padded indices (offset by 1 for the ghost layer)
    ip = i + 1
    jp = j + 1

    # Node coordinates (identical to xv[i], yv[j])
    X = xvp[ip]
    Y = yvp[jp]

    # Neighbor coordinates from padded arrays (always non-degenerate)
    X1 = xvp[ip - 1]
    X2 = xvp[ip + 1]
    Y1 = yvp[jp - 1]
    Y2 = yvp[jp + 1]

    # Build 9 deformed grid positions (node + dt*velocity)
    # Ordering: (j-row, i-col) → linear index
    #  1=(j-1,i-1) 2=(j-1,i) 3=(j-1,i+1)
    #  4=(j  ,i-1) 5=(j  ,i) 6=(j  ,i+1)
    #  7=(j+1,i-1) 8=(j+1,i) 9=(j+1,i+1)
    cx = (
        dt * vxp[ip - 1, jp - 1] + X1,  # 1
        dt * vxp[ip, jp - 1] + X,   # 2
        dt * vxp[ip + 1, jp - 1] + X2,  # 3
        dt * vxp[ip - 1, jp] + X1,  # 4
        dt * vxp[ip, jp] + X,   # 5
        dt * vxp[ip + 1, jp] + X2,  # 6
        dt * vxp[ip - 1, jp + 1] + X1,  # 7
        dt * vxp[ip, jp + 1] + X,   # 8
        dt * vxp[ip + 1, jp + 1] + X2,  # 9
    )

    cy = (
        dt * vyp[ip - 1, jp - 1] + Y1,  # 1
        dt * vyp[ip, jp - 1] + Y1,  # 2
        dt * vyp[ip + 1, jp - 1] + Y1,  # 3
        dt * vyp[ip - 1, jp] + Y,   # 4
        dt * vyp[ip, jp] + Y,   # 5
        dt * vyp[ip + 1, jp] + Y,   # 6
        dt * vyp[ip - 1, jp + 1] + Y2,  # 7
        dt * vyp[ip, jp + 1] + Y2,  # 8
        dt * vyp[ip + 1, jp + 1] + Y2,  # 9
    )

    cz = (
        dt * vzp[ip - 1, jp - 1] + topop[ip - 1, jp - 1],  # 1
        dt * vzp[ip, jp - 1] + topop[ip, jp - 1],  # 2
        dt * vzp[ip + 1, jp - 1] + topop[ip + 1, jp - 1],  # 3
        dt * vzp[ip - 1, jp] + topop[ip - 1, jp],  # 4
        dt * vzp[ip, jp] + topop[ip, jp],  # 5
        dt * vzp[ip + 1, jp] + topop[ip + 1, jp],  # 6
        dt * vzp[ip - 1, jp + 1] + topop[ip - 1, jp + 1],  # 7
        dt * vzp[ip, jp + 1] + topop[ip, jp + 1],  # 8
        dt * vzp[ip + 1, jp + 1] + topop[ip + 1, jp + 1],  # 9
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

    # Updated target position with background strain (if needed)
    Xt = X + dt * Exx * X
    Yt = Y + dt * Eyy * Y

    # Search through the 16 triangles
    Z = zero(Float64)
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

    if !found
        # Fallback: use the center-node advected value
        Z = cz[5]
    end

    advected[i, j] = Z

    return nothing
end

"""
    _interpolate_triangle(cx, cy, cz, tri, xp, yp; tol=1e-10)

Check if point (xp, yp) lies inside the triangle defined by indices `tri`
into coordinate arrays `(cx, cy, cz)`, and compute the barycentric
interpolation of the z-coordinate.

# Returns
- `(true, z_interpolated)` if the point is inside the triangle
- `(false, 0.0)` otherwise
"""
@inline function _interpolate_triangle(
        cx::NTuple, cy::NTuple, cz::NTuple,
        tri::NTuple{3, Int}, xp, yp;
        tol = 1.0e-10,
    )
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
    if S > A * (1.0 + tol)
        return false, 0.0
    end

    # Normalize barycentric coordinates
    if S > 0
        la /= S
        lb /= S
        lc /= S
    else
        la = lb = lc = 1.0 / 3.0
    end

    # Interpolate z
    fp = la * cz[ia] + lb * cz[ib] + lc * cz[ic]

    return true, fp
end

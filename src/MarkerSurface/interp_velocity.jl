"""
    interpolate_velocity_to_surface_vertices!(surf::MarkerSurface, V, grid_vxi)

Interpolate the 3D velocity field `V = (Vx, Vy, Vz)` onto the free surface
nodes. Each surface node at position `(xv[i], yv[j], topo[i,j])` receives
trilinearly interpolated velocity values.

# Arguments
- `surf` : the `MarkerSurface`
- `V`    : tuple `(Vx, Vy, Vz)` of 3D velocity arrays
- `grid_vxi` : tuple of component grids `(grid_vx, grid_vy, grid_vz)`, where
  each component grid is its own `(x, y, z)` coordinate tuple
"""
function interpolate_velocity_to_surface_vertices!(
        surf::MarkerSurface, V::NTuple{3, AbstractArray{T, 3}},
        grid_vxi::NTuple{3, Any},
    ) where {T}
    _validate_surface_velocity_layout(V, grid_vxi)
    Tc = eltype(surf.topo)
    grid_vxi = recast_grid(grid_vxi, Tc)
    nx1 = length(surf.xv)
    ny1 = length(surf.yv)

    topo = surf.topo
    svx = surf.vx
    svy = surf.vy
    svz = surf.vz
    sxv = surf.xv
    syv = surf.yv
    Vx, Vy, Vz = V
    grid_vx, grid_vy, grid_vz = grid_vxi

    launch!(
        ka_backend(topo), _interpolate_velocity_kernel!, (nx1, ny1),
        svx, svy, svz, topo, sxv, syv, Vx, Vy, Vz,
        grid_vx..., grid_vy..., grid_vz...
    )

    # MPI: under z-decomposition each rank only interpolated velocities for
    # surface nodes within its own z-slab; combine them across the z-column.
    reduce_surface_velocity_z!(surf, grid_vz[3])

    return nothing
end

@kernel function _interpolate_velocity_kernel!(
        svx, svy, svz, topo, sxv, syv, Vx, Vy, Vz,
        xgx, ygx, zgx, xgy, ygy, zgy, xgz, ygz, zgz,
    )
    i, j = @index(Global, NTuple)
    z_surf = topo[i, j]
    x_s = sxv[i]
    y_s = syv[j]

    # Interpolate each velocity component
    svx[i, j] = _interp_vel_component(Vx, xgx, ygx, zgx, x_s, y_s, z_surf)
    svy[i, j] = _interp_vel_component(Vy, xgy, ygy, zgy, x_s, y_s, z_surf)
    svz[i, j] = _interp_vel_component(Vz, xgz, ygz, zgz, x_s, y_s, z_surf)
end

"""
    _validate_surface_velocity_layout(V, grid_vxi)

Throw unless each `grid_vxi[d]` is an `(x, y, z)` tuple of usable coordinate
vectors whose lengths match `size(V[d])` — the interpolation reads `V[d]` at cell
indices looked up in `grid_vxi[d]`.
"""
function _validate_surface_velocity_layout(V, grid_vxi)
    for d in 1:3
        grid = grid_vxi[d]
        grid isa Tuple && length(grid) == 3 ||
            throw(ArgumentError("grid_vxi[$d] must be an (x, y, z) coordinate tuple"))
        all(length(coord) ≥ 2 for coord in grid) ||
            throw(ArgumentError("each coordinate vector in grid_vxi[$d] must contain at least two entries"))
        size(V[d]) == Tuple(length.(grid)) ||
            throw(DimensionMismatch("size(V[$d]) = $(size(V[d])) does not match grid_vxi[$d] lengths $(Tuple(length.(grid)))"))
    end
    return nothing
end

"""
    _interp_vel_component(Vcomp, xg, yg, zg, x, y, z)

Interpolate a single velocity component at point `(x, y, z)` by trilinear
interpolation of the 3D array `Vcomp` over its own vertex coordinates
`(xg, yg, zg)`. Points outside the grid clamp to the boundary cell.
"""
@inline function _interp_vel_component(
        Vcomp::AbstractArray{T, 3}, xg, yg, zg,
        x, y, z,
    ) where {T}
    i = _find_cell_1d(xg, x)
    j = _find_cell_1d(yg, y)
    k = _find_cell_1d(zg, z)

    # Clamp to valid range
    i = clamp(i, 1, length(xg) - 1)
    j = clamp(j, 1, length(yg) - 1)
    k = clamp(k, 1, length(zg) - 1)

    wx = (x - xg[i]) / (xg[i + 1] - xg[i])
    wy = (y - yg[j]) / (yg[j + 1] - yg[j])
    wz = (z - zg[k]) / (zg[k + 1] - zg[k])

    wx = clamp(wx, zero(T), one(T))
    wy = clamp(wy, zero(T), one(T))
    wz = clamp(wz, zero(T), one(T))

    return _trilinear(Vcomp, i, j, k, wx, wy, wz)
end

@inline function _interp_vel_component(
        Vcomp::AbstractArray{T, 3},
        xg::AbstractRange, yg::AbstractRange, zg::AbstractRange,
        x, y, z,
    ) where {T}
    i, wx = _uniform_cell_weight(xg, x)
    j, wy = _uniform_cell_weight(yg, y)
    k, wz = _uniform_cell_weight(zg, z)
    return _trilinear(Vcomp, i, j, k, wx, wy, wz)
end

"""
    _uniform_cell_weight(coords, val)

Cell index and interpolation weight of `val` within the uniformly spaced
`coords`, computed from `first`/`step` instead of a search. Values outside the
range clamp to the boundary cell.
"""
@inline function _uniform_cell_weight(coords::AbstractRange{T}, val) where {T}
    q = (convert(T, val) - convert(T, first(coords))) / convert(T, step(coords))
    q = clamp(q, zero(T), convert(T, length(coords) - 1))
    i = clamp(floor(Int, q) + 1, 1, length(coords) - 1)
    return i, clamp(q - convert(T, i - 1), zero(T), one(T))
end

"""
    _find_cell_1d(coords, val)

Find index `k` such that `coords[k] <= val < coords[k+1]`.
Returns 0 if `val < coords[1]`, or `length(coords)` if `val >= coords[end]`.
"""
@inline function _find_cell_1d(coords, val)
    n = length(coords)
    n < 2 && return 1
    val < coords[1] && return 0
    val >= coords[end] && return n
    # binary search: largest k with coords[k] <= val
    lo = 1
    hi = n - 1
    while lo < hi
        mid = (lo + hi + 1) >> 1
        if coords[mid] <= val
            lo = mid
        else
            hi = mid - 1
        end
    end
    return lo
end

@inline function _find_cell_1d(coords::AbstractRange{T}, val) where {T}
    n = length(coords)
    n < 2 && return 1
    val = convert(T, val)
    x0 = convert(T, first(coords))
    dx = convert(T, step(coords))
    val < x0 && return 0
    val >= last(coords) && return n

    i = clamp(floor(Int, (val - x0) / dx) + 1, 1, n - 1)
    i = ifelse(val < coords[i], i - 1, i)
    i = ifelse(val >= coords[i + 1], i + 1, i)
    return i
end

"""
    _trilinear(F, i, j, k, wx, wy, wz)

Trilinear interpolation of 3D field `F` at cell `(i,j,k)` with weights `(wx,wy,wz)`.
"""
@inline function _trilinear(F::AbstractArray{T, 3}, i, j, k, wx, wy, wz) where {T}
    nx, ny, nz = size(F)
    i2 = min(i + 1, nx)
    j2 = min(j + 1, ny)
    k2 = min(k + 1, nz)

    c000 = F[i, j, k]
    c100 = F[i2, j, k]
    c010 = F[i, j2, k]
    c110 = F[i2, j2, k]
    c001 = F[i, j, k2]
    c101 = F[i2, j, k2]
    c011 = F[i, j2, k2]
    c111 = F[i2, j2, k2]

    c00 = c000 * (1 - wx) + c100 * wx
    c01 = c001 * (1 - wx) + c101 * wx
    c10 = c010 * (1 - wx) + c110 * wx
    c11 = c011 * (1 - wx) + c111 * wx

    c0 = c00 * (1 - wy) + c10 * wy
    c1 = c01 * (1 - wy) + c11 * wy

    return c0 * (1 - wz) + c1 * wz
end

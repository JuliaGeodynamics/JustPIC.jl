"""
    interpolate_velocity_to_surface_vertices!(surf::MarkerSurface, V, xvi)

Interpolate the 3D velocity field `V = (Vx, Vy, Vz)` onto the free surface
nodes. Each surface node at position `(xv[i], yv[j], topo[i,j])` receives
trilinearly interpolated velocity values.

# Arguments
- `surf` : the `MarkerSurface`
- `V`    : tuple `(Vx, Vy, Vz)` of 3D velocity arrays
- `xvi`  : tuple `(xv, yv, zv)` of 1D vertex coordinate arrays
"""
function interpolate_velocity_to_surface_vertices!(
        surf::MarkerSurface, V::NTuple{3, AbstractArray{T, 3}},
        xvi::NTuple{3, Any},
    ) where {T}
    nx1 = length(surf.xv)
    ny1 = length(surf.yv)

    topo = surf.topo
    svx = surf.vx
    svy = surf.vy
    svz = surf.vz
    sxv = surf.xv
    syv = surf.yv
    Vx, Vy, Vz = V

    @parallel (1:nx1, 1:ny1) _interpolate_velocity_kernel!(
        svx, svy, svz, topo, sxv, syv, Vx, Vy, Vz, xvi...
    )

    return nothing
end

@parallel_indices (i, j) function _interpolate_velocity_kernel!(
        svx, svy, svz, topo, sxv, syv, Vx, Vy, Vz, xg, yg, zg
    )
    z_surf = topo[i, j]
    x_s = sxv[i]
    y_s = syv[j]

    # Interpolate each velocity component
    svx[i, j] = _interp_vel_component(Vx, xg, yg, zg, x_s, y_s, z_surf)
    svy[i, j] = _interp_vel_component(Vy, xg, yg, zg, x_s, y_s, z_surf)
    svz[i, j] = _interp_vel_component(Vz, xg, yg, zg, x_s, y_s, z_surf)

    return nothing
end

"""
    _interp_vel_component(Vcomp, xvi, x, y, z)

Interpolate a single velocity component at point `(x, y, z)` from the 3D grid
using trilinear interpolation.
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
    for k in 1:(n - 1)
        if coords[k] <= val < coords[k + 1]
            return k
        end
    end
    return n - 1
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

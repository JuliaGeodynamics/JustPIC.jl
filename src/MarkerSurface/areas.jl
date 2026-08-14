"""
    compute_rock_fraction!(ratios, surf::MarkerSurface, xvi, dxi)

Compute the rock fraction (fraction of each cell volume below the free surface)
at all staggered-grid positions and store them in `ratios`.

This is the 3D equivalent of `compute_rock_fraction!(ratios, chain::MarkerChain, xvi, dxi)`.
The `ratios` struct must have fields `.center`, `.vertex`, `.Vx`, `.Vy`, `.Vz`,
`.xy`, `.yz`, and `.xz`.

# Arguments
- `ratios` : struct with center, vertex, face, and edge arrays
- `surf`   : the `MarkerSurface`
- `xvi`    : tuple `(xv, yv, zv)` of 1D vertex coordinate arrays
- `dxi`    : tuple `(dx, dy, dz)` of grid spacings (kept for API consistency
             with the 2D version; the 3D kernel uses `xvi` directly)
"""
function compute_rock_fraction!(ratios, surf::MarkerSurface, xvi, dxi)
    compute_volume_below_surface!(ratios.center, surf, xvi, Val(false), Val(false), Val(false))
    compute_volume_below_surface!(ratios.vertex, surf, xvi, Val(true), Val(true), Val(true))
    compute_volume_below_surface!(ratios.Vx, surf, xvi, Val(true), Val(false), Val(false))
    compute_volume_below_surface!(ratios.Vy, surf, xvi, Val(false), Val(true), Val(false))
    compute_volume_below_surface!(ratios.Vz, surf, xvi, Val(false), Val(false), Val(true))
    compute_volume_below_surface!(ratios.xy, surf, xvi, Val(true), Val(true), Val(false))
    compute_volume_below_surface!(ratios.yz, surf, xvi, Val(false), Val(true), Val(true))
    compute_volume_below_surface!(ratios.xz, surf, xvi, Val(true), Val(false), Val(true))

    return nothing
end

function compute_volume_below_surface!(ratio, surf, xvi, px, py, pz)
    xv, yv, zv = recast_grid(xvi, eltype(ratio))
    launch!(
        ka_backend(ratio), _compute_volume_below_surface!, size(ratio),
        ratio, surf.topo, xv, yv, zv, px, py, pz,
    )
    return nothing
end

@kernel function _compute_volume_below_surface!(
        rock_fraction, topo, xv, yv, zv, px, py, pz,
    )
    i, j, k = @index(Global, NTuple)
    rock_fraction[i, j, k] = _control_volume_rock_fraction(
        topo, xv, yv, zv, i, j, k, px, py, pz,
    )
end

@inline _control_bounds(xv, i, ::Val{false}) = (xv[i], xv[i + 1])

@inline function _control_bounds(xv, i, ::Val{true})
    lo = i == 1 ? xv[1] : (xv[i - 1] + xv[i]) / 2
    hi = i == length(xv) ? xv[end] : (xv[i] + xv[i + 1]) / 2
    return lo, hi
end

@inline _surface_cell_range(i, n, ::Val{false}) = (i, i)
@inline _surface_cell_range(i, n, ::Val{true}) = (max(i - 1, 1), min(i, n))

@inline _surface_quadrant_range(a, i, ::Val{false}) = (1, 2)
@inline _surface_quadrant_range(a, i, ::Val{true}) = a < i ? (2, 2) : (1, 1)

@inline function _control_volume_rock_fraction(
        topo, xv, yv, zv, i, j, k, px, py, pz,
    )
    xlo, xhi = _control_bounds(xv, i, px)
    ylo, yhi = _control_bounds(yv, j, py)
    zlo, zhi = _control_bounds(zv, k, pz)
    vcell = (xhi - xlo) * (yhi - ylo) * (zhi - zlo)
    nx, ny = length(xv) - 1, length(yv) - 1
    amin, amax = _surface_cell_range(i, nx, px)
    bmin, bmax = _surface_cell_range(j, ny, py)
    ratio = zero(vcell)

    for b in bmin:bmax, a in amin:amax
        qxmin, qxmax = _surface_quadrant_range(a, i, px)
        qymin, qymax = _surface_quadrant_range(b, j, py)
        for qy in qymin:qymax, qx in qxmin:qxmax
            ratio += _quadrant_rock_fraction(topo, xv, yv, a, b, qx, qy, vcell, zlo, zhi)
        end
    end

    return clamp(ratio, zero(ratio), one(ratio))
end

@inline function _quadrant_rock_fraction(topo, xv, yv, a, b, qx, qy, vcell, zlo, zhi)
    x0, x1 = xv[a], xv[a + 1]
    y0, y1 = yv[b], yv[b + 1]
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    z00 = topo[a, b]
    z10 = topo[a + 1, b]
    z01 = topo[a, b + 1]
    z11 = topo[a + 1, b + 1]
    zxm0, zxm1 = (z00 + z10) / 2, (z01 + z11) / 2
    z0ym, z1ym = (z00 + z01) / 2, (z10 + z11) / 2
    zc = (z00 + z10 + z01 + z11) / 4

    if qx == 1 && qy == 1
        return _triangle_rock_fraction(x0, y0, z00, xm, y0, zxm0, xm, ym, zc, vcell, zlo, zhi) +
            _triangle_rock_fraction(x0, y0, z00, xm, ym, zc, x0, ym, z0ym, vcell, zlo, zhi)
    elseif qx == 2 && qy == 1
        return _triangle_rock_fraction(x1, y0, z10, x1, ym, z1ym, xm, ym, zc, vcell, zlo, zhi) +
            _triangle_rock_fraction(x1, y0, z10, xm, ym, zc, xm, y0, zxm0, vcell, zlo, zhi)
    elseif qx == 2 && qy == 2
        return _triangle_rock_fraction(x1, y1, z11, xm, y1, zxm1, xm, ym, zc, vcell, zlo, zhi) +
            _triangle_rock_fraction(x1, y1, z11, xm, ym, zc, x1, ym, z1ym, vcell, zlo, zhi)
    end
    return _triangle_rock_fraction(x0, y1, z01, x0, ym, z0ym, xm, ym, zc, vcell, zlo, zhi) +
        _triangle_rock_fraction(x0, y1, z01, xm, ym, zc, xm, y1, zxm1, vcell, zlo, zhi)
end

"""
    is_surface_above_cell(zmin_surface, cell::BBox{3})

Check whether the topographic surface is entirely above (or at) the cell top.
"""
@inline function is_surface_above_cell(zmin_surface, cell::BBox{3})
    ztop = cell.origin[3] + cell.d
    return GridGeometryUtils.geq_r(zmin_surface, ztop)
end

"""
    is_surface_below_cell(zmax_surface, cell::BBox{3})

Check whether the topographic surface is entirely below (or at) the cell bottom.
"""
@inline function is_surface_below_cell(zmax_surface, cell::BBox{3})
    zbot = cell.origin[3]
    return GridGeometryUtils.leq_r(zmax_surface, zbot)
end

"""
    cell_rock_volume(cz1, cz2, cz3, cz4, cell::BBox{3})

Compute the rock fraction of a 3D cell based on the topographic surface heights
at the four cell corners. Returns a value clamped to [0, 1].
"""
@inline function cell_rock_volume(cz1, cz2, cz3, cz4, cell::BBox{3, T}) where {T}
    zmin_surf = min(cz1, cz2, cz3, cz4)
    zmax_surf = max(cz1, cz2, cz3, cz4)

    A = if is_surface_above_cell(zmin_surf, cell)
        one(T)
    elseif is_surface_below_cell(zmax_surf, cell)
        zero(T)
    else
        clamp(
            _intersecting_rock_fraction(cz1, cz2, cz3, cz4, cell),
            zero(T), one(T),
        )
    end

    return A
end

"""
    _intersecting_rock_fraction(cz1, cz2, cz3, cz4, cell::BBox{3})

Compute the rock fraction via the 4-triangle prism intersection algorithm. Uses `volume(cell)` from GGU.
"""
@inline function _intersecting_rock_fraction(cz1, cz2, cz3, cz4, cell::BBox{3})
    # Cell geometry from GGU BBox
    ox, oy = cell.origin[1], cell.origin[2]
    dx, dy = cell.l, cell.h
    zbot = cell.origin[3]
    ztop = zbot + cell.d
    vcell = volume(cell)

    xleft, xright = ox, ox + dx
    yfront, yback = oy, oy + dy

    # Midpoint topography
    cz5 = (cz1 + cz2 + cz3 + cz4) / 4

    # Coordinates of corners + midpoint
    cx = (xleft, xright, xleft, xright, (xleft + xright) / 2)
    cy = (yfront, yfront, yback, yback, (yfront + yback) / 2)
    cz = (cz1, cz2, cz3, cz4, cz5)

    # 4-triangle tessellation of the cell top-face
    # corners: 1=(left,front), 2=(right,front), 3=(left,back), 4=(right,back), 5=center
    tria = ((1, 2, 5), (2, 4, 5), (4, 3, 5), (3, 1, 5))

    rock_ratio = zero(vcell)
    for tri in tria
        rock_ratio += _intersect_triangular_prism(cx, cy, cz, tri, vcell, zbot, ztop)
    end

    return rock_ratio
end

"""
    _intersect_triangular_prism(cx, cy, cz, tri, vcell, bot, top)

Compute the volume of a triangular prism (with a slanted top surface defined
by the topography) that is inside the cell bounded by `[bot, top]` in z,
normalized by `vcell`.

# Returns
Rock fraction contributed by this triangle (in [0, 0.25] for a 4-triangle cell).
"""
@inline function _intersect_triangular_prism(
        cx::NTuple, cy::NTuple, cz::NTuple,
        tri::NTuple{3, Int}, vcell,
        bot, top,
    )
    ia, ib, ic = tri
    xa, ya, za = cx[ia], cy[ia], cz[ia]
    xb, yb, zb = cx[ib], cy[ib], cz[ib]
    xc, yc, zc = cx[ic], cy[ic], cz[ic]

    return _triangle_rock_fraction(xa, ya, za, xb, yb, zb, xc, yc, zc, vcell, bot, top)
end

@inline function _triangle_rock_fraction(
        xa, ya, za, xb, yb, zb, xc, yc, zc, vcell, bot, top,
    )

    # z-range of the surface
    zmin = min(za, zb, zc)
    zmax = max(za, zb, zc)

    # Empty cell: surface entirely below cell bottom
    zmax ≤ bot && return zero(vcell)

    # Full prism: normalize this triangle's horizontal area by the control volume.
    zmin ≥ top && return abs((xa - xc) * (yb - yc) - (xb - xc) * (ya - yc)) *
        (top - bot) / (2 * vcell)

    # Volume above bottom plane
    tol = (top - bot) * convert(typeof(za), 1.0e-6)
    vbot = _prism_volume_above_level(xa, ya, za, xb, yb, zb, xc, yc, zc, bot, tol)

    # Volume above top plane
    vtop = ifelse(
        zmax > top,
        _prism_volume_above_level(xa, ya, za, xb, yb, zb, xc, yc, zc, top, tol),
        zero(vcell),
    )

    # Volume inside cell = volume above bottom - volume above top
    return (vbot - vtop) / (2 * vcell)
end

"""
    _prism_volume_above_level(xa,ya,za, xb,yb,zb, xc,yc,zc, level)

Compute double the volume of the triangular prism above a horizontal `level` plane.
"""
@inline function _prism_volume_above_level(
        xa, ya, za, xb, yb, zb, xc, yc, zc, level, tol,
    )
    # Intersect edges with level
    xab, yab, zab = _intersect_edge(xa, ya, za, xb, yb, zb, level, tol)
    xbc, ybc, zbc = _intersect_edge(xb, yb, zb, xc, yc, zc, level, tol)
    xca, yca, zca = _intersect_edge(xc, yc, zc, xa, ya, za, level, tol)

    vol = _get_volume_prism(xa, ya, za, xab, yab, zab, xca, yca, zca, level)
    vol += _get_volume_prism(xb, yb, zb, xbc, ybc, zbc, xab, yab, zab, level)
    vol += _get_volume_prism(xc, yc, zc, xca, yca, zca, xbc, ybc, zbc, level)
    vol += _get_volume_prism(xab, yab, zab, xbc, ybc, zbc, xca, yca, zca, level)

    return vol
end

"""
    _intersect_edge(x1,y1,z1, x2,y2,z2, level)

Find the intersection point of edge (p1→p2) with the horizontal plane `z=level`.
Clamps the intersection to lie within the edge's z-range.
"""
@inline function _intersect_edge(x1, y1, z1, x2, y2, z2, level, tol)
    zp = level
    zp = max(zp, min(z1, z2))
    zp = min(zp, max(z1, z2))

    dz = z2 - z1
    if abs(dz) < tol
        w = zero(dz)
    else
        w = (zp - z1) / dz
    end
    w = clamp(w, zero(w), one(w))

    xp = x1 + w * (x2 - x1)
    yp = y1 + w * (y2 - y1)

    return xp, yp, zp
end

"""
    _get_volume_prism(x1,y1,z1, x2,y2,z2, x3,y3,z3, level)

Compute double the volume of a prism above `level`
"""
@inline function _get_volume_prism(x1, y1, z1, x2, y2, z2, x3, y3, z3, level)
    avg_z = (z1 + z2 + z3) / 3
    if avg_z > level
        area2 = abs((x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3))
        return (avg_z - level) * area2
    end
    return zero(avg_z)
end

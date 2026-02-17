# ══════════════════════════════════════════════════════════════════════════ #
#       compute_rock_fraction!  —  3D MarkerSurface dispatch                #
# ══════════════════════════════════════════════════════════════════════════ #
#
# This is the 3D counterpart of `compute_rock_fraction!` in
# `MarkerChain/areas.jl` (2D). Both share the same function name so that
# user code can call `compute_rock_fraction!(ratios, surface, xvi, dxi)`
# regardless of dimensionality.
#
# The `ratios` argument is duck-typed: it must expose the same fields as
# JustRelax's `RockRatio` struct (`.center`, `.vertex`, `.Vx`, `.Vy`,
# `.Vz`, `.xy`, `.yz`, `.xz`).
#
# Cell-center rock fractions are computed exactly via the triangular-prism
# intersection algorithm (LaMEM's `FreeSurfGetAirPhaseRatio`), using
# GridGeometryUtils (GGU) types for cell geometry—mirroring how the 2D
# `areas.jl` uses `Segment`, `Rectangle`, and `cell_rock_area`:
#   • `BBox{3}` for cell geometry  (cf. `Rectangle` in 2D)
#   • `volume()` for cell volume   (cf. `area()` in 2D)
#   • `cell_rock_volume()` helper  (cf. `cell_rock_area()` in 2D)
#   • `is_surface_above_cell()` / `is_surface_below_cell()`
#       (cf. `is_chain_above_cell` / `is_chain_below_cell` in 2D)
#
# Staggered-node values (vertex, velocity, shear-stress) are derived by
# averaging neighbouring center values — standard staggered-grid
# interpolation, consistent with how `update_rock_ratio!` operates in JR.
# ══════════════════════════════════════════════════════════════════════════ #

"""
    compute_rock_fraction!(ratios, surf::MarkerSurface, xvi, dxi)

Compute the rock fraction (fraction of each cell volume below the free surface)
at all staggered-grid positions and store them in `ratios`.

This is the 3D equivalent of `compute_rock_fraction!(ratios, chain::MarkerChain, xvi, dxi)`.
The `ratios` struct must have fields `.center`, `.vertex`, `.Vx`, `.Vy`, `.Vz`,
`.xy`, `.yz`, `.xz` (e.g. a `RockRatio` from JustRelax).

# Arguments
- `ratios` : struct with staggered-grid arrays (e.g. `RockRatio`)
- `surf`   : the `MarkerSurface`
- `xvi`    : tuple `(xv, yv, zv)` of 1D vertex coordinate arrays
- `dxi`    : tuple `(dx, dy, dz)` of grid spacings (kept for API consistency
             with the 2D version; the 3D kernel uses `xvi` directly)
"""
function compute_rock_fraction!(ratios, surf::MarkerSurface, xvi, dxi)
    # 1. Cell-center rock fractions (exact prism intersection, cf. cell_rock_area in 2D)
    compute_volume_below_surface_centers!(ratios.center, surf, xvi)

    # 2. Vertex values — average of up to 8 neighbouring centers
    @parallel (@idx size(ratios.vertex)) _avg_center_to_vertex_3D!(
        ratios.vertex, ratios.center
    )

    # 3. Velocity nodes — average of 2 neighbours along the staggered direction
    @parallel (@idx size(ratios.Vx)) _avg_center_to_face_x!(ratios.Vx, ratios.center)
    @parallel (@idx size(ratios.Vy)) _avg_center_to_face_y!(ratios.Vy, ratios.center)
    @parallel (@idx size(ratios.Vz)) _avg_center_to_face_z!(ratios.Vz, ratios.center)

    # 4. Shear-stress nodes — average of 4 neighbours in the plane
    @parallel (@idx size(ratios.xy)) _avg_center_to_edge_xy!(ratios.xy, ratios.center)
    @parallel (@idx size(ratios.yz)) _avg_center_to_edge_yz!(ratios.yz, ratios.center)
    @parallel (@idx size(ratios.xz)) _avg_center_to_edge_xz!(ratios.xz, ratios.center)

    return nothing
end

# ──────────────────────────────────────────────────────────────────────────── #
#       Cell-center computation (prism intersection, using GGU types)        #
# ──────────────────────────────────────────────────────────────────────────── #

function compute_volume_below_surface_centers!(ratio_center, surf, xvi)
    topo = surf.topo
    xv, yv, zv = xvi
    nx = length(xv) - 1
    ny = length(yv) - 1
    nz = length(zv) - 1
    @parallel (1:nx, 1:ny, 1:nz) _compute_volume_below_surface_center!(
        ratio_center, topo, xv, yv, zv
    )
    return nothing
end

@parallel_indices (i, j, k) function _compute_volume_below_surface_center!(
        rock_fraction, topo, xv, yv, zv
    )
    # Cell geometry via GridGeometryUtils.BBox{3}  (cf. Rectangle in 2D areas.jl)
    ox = xv[i]
    oy = yv[j]
    oz = zv[k]
    dx = xv[i + 1] - ox
    dy = yv[j + 1] - oy
    dz = zv[k + 1] - oz
    cell = BBox((ox, oy, oz), dx, dy, dz)

    # Topography at the 4 corners
    cz1 = topo[i, j]  # front-left
    cz2 = topo[i + 1, j]  # front-right
    cz3 = topo[i, j + 1]  # back-left
    cz4 = topo[i + 1, j + 1]  # back-right

    rock_fraction[i, j, k] = cell_rock_volume(cz1, cz2, cz3, cz4, cell)

    return nothing
end

# ──────────────────────────────────────────────────────────────────────────── #
#               Center → staggered averaging kernels (3D)                    #
# ──────────────────────────────────────────────────────────────────────────── #

@parallel_indices (i, j, k) function _avg_center_to_vertex_3D!(vertex, center)
    nx, ny, nz = size(center)
    s = 0.0
    ω = 0
    for dk in 0:1, dj in 0:1, di in 0:1
        ii = i - 1 + di
        jj = j - 1 + dj
        kk = k - 1 + dk
        if 1 ≤ ii ≤ nx && 1 ≤ jj ≤ ny && 1 ≤ kk ≤ nz
            s += center[ii, jj, kk]
            ω += 1
        end
    end
    vertex[i, j, k] = ω > 0 ? s / ω : 0.0
    return nothing
end

@parallel_indices (i, j, k) function _avg_center_to_face_x!(Vx, center)
    nx, ny, nz = size(center)
    s = 0.0
    ω = 0
    for di in 0:1
        ii = i - 1 + di
        if 1 ≤ ii ≤ nx
            s += center[ii, j, k]
            ω += 1
        end
    end
    Vx[i, j, k] = ω > 0 ? s / ω : 0.0
    return nothing
end

@parallel_indices (i, j, k) function _avg_center_to_face_y!(Vy, center)
    nx, ny, nz = size(center)
    s = 0.0
    ω = 0
    for dj in 0:1
        jj = j - 1 + dj
        if 1 ≤ jj ≤ ny
            s += center[i, jj, k]
            ω += 1
        end
    end
    Vy[i, j, k] = ω > 0 ? s / ω : 0.0
    return nothing
end

@parallel_indices (i, j, k) function _avg_center_to_face_z!(Vz, center)
    nx, ny, nz = size(center)
    s = 0.0
    ω = 0
    for dk in 0:1
        kk = k - 1 + dk
        if 1 ≤ kk ≤ nz
            s += center[i, j, kk]
            ω += 1
        end
    end
    Vz[i, j, k] = ω > 0 ? s / ω : 0.0
    return nothing
end

@parallel_indices (i, j, k) function _avg_center_to_edge_xy!(xy, center)
    nx, ny, nz = size(center)
    s = 0.0
    ω = 0
    for dj in 0:1, di in 0:1
        ii = i - 1 + di
        jj = j - 1 + dj
        if 1 ≤ ii ≤ nx && 1 ≤ jj ≤ ny && 1 ≤ k ≤ nz
            s += center[ii, jj, k]
            ω += 1
        end
    end
    xy[i, j, k] = ω > 0 ? s / ω : 0.0
    return nothing
end

@parallel_indices (i, j, k) function _avg_center_to_edge_yz!(yz, center)
    nx, ny, nz = size(center)
    s = 0.0
    ω = 0
    for dk in 0:1, dj in 0:1
        jj = j - 1 + dj
        kk = k - 1 + dk
        if 1 ≤ i ≤ nx && 1 ≤ jj ≤ ny && 1 ≤ kk ≤ nz
            s += center[i, jj, kk]
            ω += 1
        end
    end
    yz[i, j, k] = ω > 0 ? s / ω : 0.0
    return nothing
end

@parallel_indices (i, j, k) function _avg_center_to_edge_xz!(xz, center)
    nx, ny, nz = size(center)
    s = 0.0
    ω = 0
    for dk in 0:1, di in 0:1
        ii = i - 1 + di
        kk = k - 1 + dk
        if 1 ≤ ii ≤ nx && 1 ≤ j ≤ ny && 1 ≤ kk ≤ nz
            s += center[ii, j, kk]
            ω += 1
        end
    end
    xz[i, j, k] = ω > 0 ? s / ω : 0.0
    return nothing
end

# ──────────────────────────────────────────────────────────────────────────── #
#    Geometry helpers: cell_rock_volume (3D analogue of cell_rock_area)      #
#    cf. MarkerChain/areas.jl for the 2D equivalent pattern                  #
# ──────────────────────────────────────────────────────────────────────────── #

"""
    is_surface_above_cell(zmin_surface, cell::BBox{3})

Check whether the topographic surface is entirely above (or at) the cell top.
Analogous to `is_chain_above_cell` in 2D `MarkerChain/areas.jl`.
"""
@inline function is_surface_above_cell(zmin_surface::Real, cell::BBox{3})
    ztop = cell.origin[3] + cell.d
    return GridGeometryUtils.geq_r(zmin_surface, ztop)
end

"""
    is_surface_below_cell(zmax_surface, cell::BBox{3})

Check whether the topographic surface is entirely below (or at) the cell bottom.
Analogous to `is_chain_below_cell` in 2D `MarkerChain/areas.jl`.
"""
@inline function is_surface_below_cell(zmax_surface::Real, cell::BBox{3})
    zbot = cell.origin[3]
    return GridGeometryUtils.leq_r(zmax_surface, zbot)
end

"""
    cell_rock_volume(cz1, cz2, cz3, cz4, cell::BBox{3})

Compute the rock fraction of a 3D cell based on the topographic surface heights
at the four cell corners. Returns a value clamped to [0, 1].

This is the 3D analogue of `cell_rock_area(s::Segment, r::Rectangle)` in
`MarkerChain/areas.jl`.
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

# ──────────────────────────────────────────────────────────────────────────── #
#          Prism intersection (LaMEM algorithm, using GGU volume())          #
# ──────────────────────────────────────────────────────────────────────────── #

"""
    _intersecting_rock_fraction(cz1, cz2, cz3, cz4, cell::BBox{3})

Compute the rock fraction via the 4-triangle prism intersection algorithm
(LaMEM's `FreeSurfGetAirPhaseRatio`).  Uses `volume(cell)` from GGU.
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

    air_ratio = 1.0
    for tri in tria
        air_ratio -= _intersect_triangular_prism(cx, cy, cz, tri, vcell, zbot, ztop)
    end

    return 1.0 - air_ratio
end

"""
    _intersect_triangular_prism(cx, cy, cz, tri, vcell, bot, top; tol=1e-12)

Compute the volume of a triangular prism (with a slanted top surface defined
by the topography) that is inside the cell bounded by `[bot, top]` in z,
normalized by `vcell`.

This is the Julia equivalent of LaMEM's `IntersectTriangularPrism`.

# Returns
Rock fraction contributed by this triangle (in [0, 0.25] for a 4-triangle cell).
"""
@inline function _intersect_triangular_prism(
        cx::NTuple, cy::NTuple, cz::NTuple,
        tri::NTuple{3, Int}, vcell::Real,
        bot::Real, top::Real;
        tol::Real = 1.0e-12,
    )
    ia, ib, ic = tri
    xa, ya, za = cx[ia], cy[ia], cz[ia]
    xb, yb, zb = cx[ib], cy[ib], cz[ib]
    xc, yc, zc = cx[ic], cy[ic], cz[ic]

    dh = (top - bot) * tol

    # z-range of the surface
    zmin = min(za, zb, zc)
    zmax = max(za, zb, zc)

    # Empty cell: surface entirely below cell bottom
    zmax ≤ bot && return 0.0

    # Full cell: surface entirely above cell top
    zmin ≥ top && return 0.25  # quarter because 4 triangles per cell

    # Volume above bottom plane
    vbot = _prism_volume_above_level(xa, ya, za, xb, yb, zb, xc, yc, zc, bot, dh)

    # Volume above top plane
    vtop = 0.0
    if zmax > top
        vtop = _prism_volume_above_level(xa, ya, za, xb, yb, zb, xc, yc, zc, top, dh)
    end

    # Volume inside cell = volume above bottom - volume above top
    return (vbot - vtop) / (2.0 * vcell)
end

"""
    _prism_volume_above_level(xa,ya,za, xb,yb,zb, xc,yc,zc, level, dh)

Compute double the volume of the triangular prism above a horizontal `level` plane.
Uses the same edge-intersection approach as LaMEM's macros.
"""
@inline function _prism_volume_above_level(
        xa, ya, za, xb, yb, zb, xc, yc, zc, level, dh
    )
    # Intersect edges with level
    xab, yab, zab = _intersect_edge(xa, ya, za, xb, yb, zb, level)
    xbc, ybc, zbc = _intersect_edge(xb, yb, zb, xc, yc, zc, level)
    xca, yca, zca = _intersect_edge(xc, yc, zc, xa, ya, za, level)

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
@inline function _intersect_edge(x1, y1, z1, x2, y2, z2, level)
    zp = level
    zp = max(zp, min(z1, z2))
    zp = min(zp, max(z1, z2))

    dz = z2 - z1
    if abs(dz) < 1.0e-30
        w = 0.0
    else
        w = (zp - z1) / dz
    end
    w = clamp(w, 0.0, 1.0)

    xp = x1 + w * (x2 - x1)
    yp = y1 + w * (y2 - y1)

    return xp, yp, zp
end

"""
    _get_volume_prism(x1,y1,z1, x2,y2,z2, x3,y3,z3, level)

Compute double the volume of a prism above `level`, as done by LaMEM's GET_VOLUME_PRISM macro.
"""
@inline function _get_volume_prism(x1, y1, z1, x2, y2, z2, x3, y3, z3, level)
    avg_z = (z1 + z2 + z3) / 3.0
    if avg_z > level
        area2 = abs((x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3))
        return (avg_z - level) * area2
    end
    return 0.0
end

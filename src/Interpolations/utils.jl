@inline function parent_cell(
    p::NTuple{N,A}, di::NTuple{N,B}, xci::NTuple{N,B}
) where {N,A,B}
    ni = length.(xci)
    return ntuple(i -> min(Int((p[i] - xci[i]) ÷ di[i] + 1), ni[i]), Val(N))
end

# dimension-agnostic fully unrolled euclidean distance
@inline function distance(a::NTuple{N,T}, b::NTuple{N,T}) where {N,T}
    return distance((a[1] - b[1])^2, Base.tail(a), Base.tail(b))
end

@inline function distance(s::Number, a::NTuple{N,T}, b::NTuple{N,T}) where {N,T}
    return distance(s + (a[1] - b[1])^2, Base.tail(a), Base.tail(b))
end

@inline function distance(s::Number, a::NTuple{1,T}, b::NTuple{1,T}) where {T}
    return √(s + (a[1] - b[1])^2)
end

# check whether particle is inside the grid (includes boundary)
@inline function isinside(px::Real, py::Real, x, y)
    xmin, xmax = extrema(x)
    ymin, ymax = extrema(y)
    return @assert (px === NaN) || (py === NaN) (xmin ≤ px ≤ xmax) && (ymin ≤ py ≤ ymax)
end

@inline function isinside(px::Real, py::Real, pz::Real, x, y, z)
    xmin, xmax = extrema(x)
    ymin, ymax = extrema(y)
    zmin, zmax = extrema(z)
    return @assert (px === NaN) ||
        (py === NaN) ||
        (pz === NaN) ||
        (xmin ≤ px ≤ xmax) && (ymin ≤ py ≤ ymax) && (zmin ≤ pz ≤ zmax)
end

@inline function isinside(p::NTuple{2,T1}, x::NTuple{2,T2}) where {T1,T2}
    return isinside(p[1], p[2], x[1], x[2])
end

@inline function isinside(p::NTuple{3,T1}, x::NTuple{3,T2}) where {T1,T2}
    return isinside(p[1], p[2], p[3], x[1], x[2], x[3])
end

# normalize coordinates
@inline function normalize_coordinates(
    p::NTuple{N,A}, xi::NTuple{N,B}, di::NTuple{N,C}, idx::NTuple{N,D}
) where {N,A,B,C,D}
    return ntuple(i -> (p[i] - xi[i][idx[i]]) * inv(di[i]), Val(N))
end

# normalize coordinates
@inline function normalize_coordinates(
    p::NTuple{N,A}, xci::NTuple{N,B}, di::NTuple{N,C}
) where {N,A,B,C}
    return ntuple(i -> (p[i] - xci[i]) * inv(di[i]), Val(N))
end

# compute grid size
function grid_size(x::NTuple{N,T}) where {T,N}
    return ntuple(i -> abs(minimum(diff(x[i]))), Val(N))
end

# Get field F at the corners of a given cell
@inline function field_corners(F::AbstractArray{T,2}, idx::NTuple{2,Int64}) where {T}
    idx_x, idx_y = idx
    return (
        F[idx_x, idx_y], F[idx_x + 1, idx_y], F[idx_x, idx_y + 1], F[idx_x + 1, idx_y + 1]
    )
end

@inline function field_corners(F::AbstractArray{T,3}, idx::NTuple{3,Int64}) where {T}
    idx_x, idx_y, idx_z = idx
    return (
        F[idx_x, idx_y, idx_z],             # v000
        F[idx_x + 1, idx_y, idx_z],         # v100
        F[idx_x, idx_y + 1, idx_z],         # v010
        F[idx_x + 1, idx_y + 1, idx_z],     # v110
        F[idx_x, idx_y, idx_z + 1],         # v001
        F[idx_x + 1, idx_y, idx_z + 1],     # v101
        F[idx_x, idx_y + 1, idx_z + 1],     # v011
        F[idx_x + 1, idx_y + 1, idx_z + 1], # v111
    )
end

@inline function center_coordinate(x_i, offset_i, idx_i, dx_i)
    if idx_i == 1
        return @inbounds x_i[1] - dx_i * 0.5
    elseif offset_i == 0
        return @inbounds x_i[idx_i]
    else
        return @inbounds x_i[idx_i - 1]
    end
end

# Get field F at the corners of a given cell
@inline function field_corners(F::AbstractArray{T,2}, idx::NTuple{2,Integer}) where {T}
    idx_x, idx_y = idx
    idx_x1, idx_y1 = (idx_x, idx_y) .+ 1
    return @inbounds (
        F[idx_x, idx_y], F[idx_x1, idx_y], F[idx_x, idx_y1], F[idx_x1, idx_y1]
    )
end

@inline function field_corners(F::AbstractArray{T,3}, idx::NTuple{3,Integer}) where {T}
    idx_x, idx_y, idx_z = idx
    return @inbounds (
        F[idx_x, idx_y, idx_z],             # v000
        F[idx_x + 1, idx_y, idx_z],         # v100
        F[idx_x, idx_y + 1, idx_z],         # v010
        F[idx_x + 1, idx_y + 1, idx_z],     # v110
        F[idx_x, idx_y, idx_z + 1],         # v001
        F[idx_x + 1, idx_y, idx_z + 1],     # v101
        F[idx_x, idx_y + 1, idx_z + 1],     # v011
        F[idx_x + 1, idx_y + 1, idx_z + 1], # v111
    )
end

# Get field F at the centers of a given cell
@inline function field_centers(
    F::AbstractArray{T,2}, pxi, xi, xci_augmented, idx::NTuple{2,Integer}
) where {T}
    # unpack
    idx_x, idx_y = idx
    px, py = pxi
    xc_augmented, yc_augmented = xci_augmented
    x, y = xi[1][idx_x], xi[2][idx_x]
    # compute offsets and corrections
    offset_x = (px - x) > 0 ? 1 : 0
    offset_y = (py - y) > 0 ? 1 : 0
    # cell indices
    idx_x += offset_x
    idx_y += offset_y
    # coordinates of lower-left corner of the cell
    xc = xc_augmented[idx_x]
    yc = yc_augmented[idx_y]

    # F at the four centers
    Fi = (
        F[idx_x, idx_y], F[idx_x + 1, idx_y], F[idx_x, idx_y + 1], F[idx_x + 1, idx_y + 1]
    )

    return Fi, (xc, yc)
end

@inline function field_centers(
    F::AbstractArray{T,3}, pxi, xi, di, idx::NTuple{3,Integer}
) where {T}
    # unpack
    idx_x, idx_y, idx_z = idx
    px, py, pz = pxi
    dx, dy, dz = di
    x, y, z = xi[1][idx_x], xi[2][idx_x], xi[3][idx_x]
    # compute offsets and corrections
    offset_x = (px - x) > 0 ? 1 : 0
    offset_y = (py - y) > 0 ? 1 : 0
    offset_z = (pz - z) > 0 ? 1 : 0
    # cell indices
    idx_x += offset_x
    idx_y += offset_y
    idx_z += offset_z
    # coordinates of lower-left corner of the cell
    xc = center_coordinate(xi[1], offset_x, idx_x, dx)
    yc = center_coordinate(xi[2], offset_y, idx_y, dy)
    zc = center_coordinate(xi[3], offset_z, idx_z, dz)
    # F at the eight centers
    Fi = (
        F[idx_x, idx_y, idx_z],             # v000
        F[idx_x + 1, idx_y, idx_z],         # v100
        F[idx_x, idx_y + 1, idx_z],         # v010
        F[idx_x + 1, idx_y + 1, idx_z],     # v110
        F[idx_x, idx_y, idx_z + 1],         # v001
        F[idx_x + 1, idx_y, idx_z + 1],     # v101
        F[idx_x, idx_y + 1, idx_z + 1],     # v011
        F[idx_x + 1, idx_y + 1, idx_z + 1], # v111
    )

    return Fi, (xc, yc, zc)
end

# lower-left center coordinate
@inline function center_coordinate(pxi, xi, idx::NTuple{2,Integer})
    idx_x, idx_y = idx
    px, py = pxi
    x, y = xi[1][idx_x], xi[2][idx_x]
    idx_x = (px - x) > 0 ? idx_x + 1 : idx_x
    idx_y = (py - y) > 0 ? idx_y + 1 : idx_y

    return (
        F[idx_x, idx_y], F[idx_x + 1, idx_y], F[idx_x, idx_y + 1], F[idx_x + 1, idx_y + 1]
    )
end

@inline function center_coordinate(pxi, xi, idx::NTuple{3,Integer})
    idx_x, idx_y, idx_z = idx
    px, py, pz = pxi
    x, y, z = xi[1][idx_x], xi[2][idx_x], xi[3][idx_x]
    idx_x = (px - x) > 0 ? idx_x + 1 : idx_x
    idx_y = (py - y) > 0 ? idx_y + 1 : idx_y
    idx_z = (pz - z) > 0 ? idx_z + 1 : idx_z

    return (
        F[idx_x, idx_y, idx_z],             # v000
        F[idx_x + 1, idx_y, idx_z],         # v100
        F[idx_x, idx_y + 1, idx_z],         # v010
        F[idx_x + 1, idx_y + 1, idx_z],     # v110
        F[idx_x, idx_y, idx_z + 1],         # v001
        F[idx_x + 1, idx_y, idx_z + 1],     # v101
        F[idx_x, idx_y + 1, idx_z + 1],     # v011
        F[idx_x + 1, idx_y + 1, idx_z + 1], # v111
    )
end

@inline function particle2tuple(p::NTuple{N,AbstractArray}, ix) where {N}
    return ntuple(i -> p[i][ix], Val(N))
end

# @inline function particle2tuple(p::NTuple{N,AbstractArray}, ip::Integer, ix::NTuple{N,T}) where {N}
#     return ntuple(i -> p[i][ip, ix...], Val(N))
# end

# 2D random particle generator for regular grids
function random_particles(nxcell, x, y, dx, dy, nx, ny)
    # number of cells
    ncells = (nx - 1) * (ny - 1)
    # allocate particle coordinate arrays
    px, py = zeros(nxcell * ncells), zeros(nxcell * ncells)
    Threads.@threads for i in 1:(nx - 1)
        @inbounds for j in 1:(ny - 1)
            # lowermost-left corner of the cell
            x0, y0 = x[i], y[j]
            # cell index
            cell = i + (nx - 1) * (j - 1)
            for l in 1:nxcell
                px[(cell - 1) * nxcell + l] = rand() * dx + x0
                py[(cell - 1) * nxcell + l] = rand() * dy + y0
            end
        end
    end

    return px, py
end

# 3D random particle generator for regular grids
function random_particles(nxcell, x, y, z, dx, dy, dz, nx, ny, nz)
    # number of cells
    ncells = (nx - 1) * (ny - 1) * (nz - 1)
    # allocate particle coordinate arrays
    px, py, pz = zeros(nxcell * ncells), zeros(nxcell * ncells), zeros(nxcell * ncells)
    Threads.@threads for i in 1:(nx - 1)
        @inbounds for j in 1:(ny - 1), k in 1:(nz - 1)
            # lowermost-left corner of the cell
            x0, y0, z0 = x[i], y[j], z[k]
            # cell index
            cell = i + (nx - 1) * (j - 1) + (nx - 1) * (ny - 1) * (k - 1)
            for l in 1:nxcell
                px[(cell - 1) * nxcell + l] = rand() * dx + x0
                py[(cell - 1) * nxcell + l] = rand() * dy + y0
                pz[(cell - 1) * nxcell + l] = rand() * dz + z0
            end
        end
    end

    return px, py, pz
end

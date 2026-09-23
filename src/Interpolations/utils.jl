# dimension-agnostic fully unrolled euclidean distance
@inline function distance(a::NTuple{N, T}, b::NTuple{N, T}) where {N, T}
    return distance((a[1] - b[1])^2, Base.tail(a), Base.tail(b))
end

@inline function distance(s::Number, a::NTuple{N, T}, b::NTuple{N, T}) where {N, T}
    return distance(s + (a[1] - b[1])^2, Base.tail(a), Base.tail(b))
end

@inline function distance(s::Number, a::NTuple{1, T}, b::NTuple{1, T}) where {T}
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

@inline function isinside(p::NTuple{2, T1}, x::NTuple{2, T2}) where {T1, T2}
    return isinside(p[1], p[2], x[1], x[2])
end

@inline function isinside(p::NTuple{3, T1}, x::NTuple{3, T2}) where {T1, T2}
    return isinside(p[1], p[2], p[3], x[1], x[2], x[3])
end

# normalize coordinates
@inline function normalize_coordinates(
        p::NTuple{N, A}, xi::NTuple{N, B}, di::NTuple{N, C}, idx::NTuple{N, D}
    ) where {N, A, B, C, D}
    return ntuple(i -> (p[i] - xi[i][idx[i]]) * inv(di[i]), Val(N))
end

# normalize coordinates
@inline function normalize_coordinates(
        p::NTuple{N, A}, xci::NTuple{N, B}, di::NTuple{N, C}
    ) where {N, A, B, C}
    return ntuple(i -> (p[i] - xci[i]) * inv(di[i]), Val(N))
end

# Compute every local cell width along each grid axis.
@inline function grid_size(x::NTuple{N}) where {N}
    return ntuple(i -> diff(x[i]), Val(N))
end

@inline function local_grid_spacing(dxi::NTuple{N}, idx::NTuple{N}) where {N}
    return ntuple(i -> dxi[i][idx[i]], Val(N))
end

# Get field F at the corners of a given cell
@inline function field_corners(F::AbstractArray{T, 2}, idx::NTuple{2, Int64}) where {T}
    idx_x, idx_y = idx
    return (
        F[idx_x, idx_y], F[idx_x + 1, idx_y], F[idx_x, idx_y + 1], F[idx_x + 1, idx_y + 1],
    )
end

@inline function field_corners(F::AbstractArray{T, 3}, idx::NTuple{3, Int64}) where {T}
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

# Get field F at the corners of a given cell
@inline function field_corners(F::AbstractArray{T, 2}, idx::NTuple{2, Integer}) where {T}
    idx_x, idx_y = idx
    idx_x1, idx_y1 = (idx_x, idx_y) .+ 1
    return @inbounds (
        F[idx_x, idx_y], F[idx_x1, idx_y], F[idx_x, idx_y1], F[idx_x1, idx_y1],
    )
end

@inline function field_corners(F::AbstractArray{T, 3}, idx::NTuple{3, Integer}) where {T}
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

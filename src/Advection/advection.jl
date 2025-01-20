@inline normalised_distance(x, p, dx) = (p - x) * inv(dx)

@inline Base.@propagate_inbounds function extract_field_corners(F, i, j)
    i1, j1 = i + 1, j + 1
    b = F[i1, j]
    c = F[i, j1]
    d = F[i1, j1]
    a = F[i, j]
    return a, b, c, d
end

@inline Base.@propagate_inbounds function extract_field_corners(F, i, j, k)
    i1, j1, k1 = i + 1, j + 1, k + 1
    F000 = F[i, j, k]
    F100 = F[i1, j, k]
    F010 = F[i, j1, k]
    F110 = F[i1, j1, k]
    F001 = F[i, j, k1]
    F101 = F[i1, j, k1]
    F011 = F[i, j1, k1]
    F111 = F[i1, j1, k1]
    return F000, F100, F010, F110, F001, F101, F011, F111
end

@inline firstlast(x::Array) = first(x), last(x)
@inline firstlast(x) = extrema(x)

@inline function inner_limits(grid::NTuple{N, T}) where {N, T}
    return ntuple(Val(N)) do i
        Base.@_inline_meta
        ntuple(j -> firstlast.(grid[i])[j], Val(N))
    end
end

@generated function check_local_limits(
        local_lims::NTuple{N, T1}, p::Union{SVector{N, T2}, NTuple{N, T2}}
    ) where {N, T1, T2}
    return quote
        Base.@_inline_meta
        Base.@nexprs $N i -> !(local_lims[i][1] < p[i] < local_lims[i][2]) && return false
        return true
    end
end

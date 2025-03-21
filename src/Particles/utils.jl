@inline corner_coordinate(grid, i::Integer) = grid[i]
@inline corner_coordinate(grid, i::Integer, j::Integer) = grid[1][i], grid[2][j]
@inline corner_coordinate(grid, i::Integer, j::Integer, k::Integer) = grid[1][i], grid[2][j], grid[3][k]
@inline corner_coordinate(grid::NTuple{N, T1}, I::NTuple{N, T2}) where {T1, T2, N} =
    corner_coordinate(grid, I...)

@generated function isincell(p::NTuple{N}, xci::NTuple{N}, dxi::NTuple{N}) where {N}
    return quote
        Base.@_inline_meta
        bool = true
        Base.@nexprs $N i -> bool = bool & isincell(p[i], xci[i], dxi[i])
        return bool
    end
end
@inline isincell(px::T, xv::T, dx::T) where {T <: Real} = xv < px < xv + dx

@inline function isemptycell(
        index::AbstractArray, min_xcell::Integer, cell_indices::Vararg{Int, N}
    ) where {N}
    # first min_xcell particles
    val = 0
    for i in 1:min_xcell
        val += @inbounds @index(index[i, cell_indices...])
    end
    # early escape
    val ≥ min_xcell && return false
    # tail
    n = cellnum(index)
    for i in (min_xcell + 1):n
        val += @inbounds @index(index[i, cell_indices...])
    end
    return !(val ≥ min_xcell)
end

@parallel_indices (i) function copy_vectors!(
        dest::NTuple{N, T}, src::NTuple{N, T}
    ) where {N, T <: AbstractArray}
    for n in 1:N
        if i ≤ length(dest[n])
            @inbounds dest[n][i] = src[n][i]
        end
    end
    return nothing
end

@inline compute_dx(::Tuple{}) = ()
@inline compute_dx(grid::AbstractArray) = grid[3] - grid[2]
@inline compute_dx(grid::Tuple) = compute_dx(first(grid)), compute_dx(Base.tail(grid))...

@inline function clamp_grid_lims(grid_lims::NTuple{N}, dxi::NTuple{N}) where {N}
    clamped_limits = ntuple(Val(N)) do i
        @inline
        min_L, max_L = grid_lims[i]
        (min_L + dxi[i] * 0.01, max_L - dxi[i] * 0.01)
    end
    return clamped_limits
end

@inline function augment_lazy_grid(grid::NTuple{N}, dxi::NTuple{N}) where {N}
    xci_augmented = ntuple(Val(N)) do i
        (grid[i][1] - dxi[i]):dxi[i]:(grid[i][end] + dxi[i])
    end
    return xci_augmented
end

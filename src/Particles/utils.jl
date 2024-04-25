@inline corner_coordinate(grid, I::Integer) = grid[I]
@inline corner_coordinate(grid::NTuple{N,T1}, I::NTuple{N,T2}) where {T1,T2,N} =
    corner_coordinate(grid, I...)

@inline function corner_coordinate(grid::NTuple{N,T1}, I::Vararg{T2,N}) where {T1,T2,N}
    return ntuple(i -> grid[i][I[i]], Val(N))
end


@generated function isincell(p::NTuple{N, T}, xci::NTuple{N, T}, dxi::NTuple{N, T}) where {N,T}
    quote
        Base.@_inline_meta
        bool = true
        Base.@nexprs $N i -> bool = bool & isincell(p[i], xci[i], dxi[i])
        return bool
    end
end
@inline isincell(px::T, xv::T, dx::T) where {T<:Real} = xv < px < xv + dx

@inline function isemptycell(
    index::AbstractArray{T,N}, min_xcell::Integer, cell_indices::Vararg{Int,N}
) where {T,N}
    # first min_xcell particles
    val = 0
    for i in 1:min_xcell
        val += @inbounds @cell(index[i, cell_indices...])
    end
    # early escape
    val ≥ min_xcell && return false
    # tail
    n = cellnum(index)
    for i in (min_xcell + 1):n
        val += @inbounds @cell(index[i, cell_indices...])
    end
    return !(val ≥ min_xcell)
end

@parallel_indices (i) function copy_vectors!(
    dest::NTuple{N,T}, src::NTuple{N,T}
) where {N,T<:AbstractArray}
    for n in 1:N
        if i ≤ length(dest[n])
            @inbounds dest[n][i] = src[n][i]
        end
    end
    return nothing
end

compute_dx(grid::LinRange{T,Int64}) where {T} = grid[2] - grid[1]

@inline function compute_dx(grid::NTuple{N,LinRange{T,Int64}}) where {N,T}
    return ntuple(i -> grid[i][2] - grid[i][1], Val(N))
end

@inline function compute_dx(grid::NTuple{N,T}) where {N,T}
    return ntuple(i -> abs(minimum(diff(grid[i]))), Val(N))
end

@inline function clamp_grid_lims(grid_lims::NTuple{N,T1}, dxi::NTuple{N,T2}) where {N,T1,T2}
    clamped_limits = ntuple(Val(N)) do i
        min_L, max_L = grid_lims[i]
        (min_L + dxi[i] * 0.01, max_L - dxi[i] * 0.01)
    end
    return clamped_limits
end

@inline function augment_lazy_grid(grid::NTuple{N,T1}, dxi::NTuple{N,T2}) where {N,T1,T2}
    xci_augmented = ntuple(Val(N)) do i
        (grid[i][1] - dxi[i]):dxi[i]:(grid[i][end] + dxi[i])
    end
    return xci_augmented
end

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

@kernel function copy_vectors!(
        dest::NTuple{N, T}, src::NTuple{N, T}
    ) where {N, T <: AbstractArray}
    i = @index(Global)
    for n in 1:N
        if i ≤ length(dest[n])
            @inbounds dest[n][i] = src[n][i]
        end
    end
end

@inline compute_dx(::Tuple{}) = ()
@inline compute_dx(grid::AbstractRange) = grid[3] - grid[2]
@inline compute_dx(grid::AbstractVector) = diff(grid)
@inline compute_dx(grid::Tuple) = compute_dx(first(grid)), compute_dx(Base.tail(grid))...

function compute_dx(xi::NTuple{N, AbstractVector}, I) where {N}
    di = ntuple(Val(N)) do i
        @inline
        ii = I[i]
        x = xi[i]
        x[ii + 1] - x[ii]
    end
    return di
end

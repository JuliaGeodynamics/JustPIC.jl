# Color grid so that all elements containing the i-th node have a different color.
# During threaded assembly each thread acts upon a single color. In this way, we 
# can multithread along each color avoiding synchronisations between threads. It 
# could be extended amongst different cpus, where each cpu takes different colors
function neighbouring_cells(I::NTuple{N,T}, nxi::NTuple{N,T}) where {N,T}
    return neighbouring_cells(I..., nxi...)
end

function neighbouring_cells(i, j, nx, ny)
    idx = (
        (clamp(i - 1, 1, nx), clamp(j - 1, 1, ny)),
        (clamp(i, 1, nx), clamp(j - 1, 1, ny)),
        (clamp(i + 1, 1, nx), clamp(j - 1, 1, ny)),
        (clamp(i - 1, 1, nx), clamp(j, 1, ny)),
        (clamp(i + 1, 1, nx), clamp(j, 1, ny)),
        (clamp(i - 1, 1, nx), clamp(j + 1, 1, ny)),
        (clamp(i, 1, nx), clamp(j + 1, 1, ny)),
        (clamp(i + 1, 1, nx), clamp(j + 1, 1, ny)),
    )
    return idx
end

function neighbouring_cells(i, j, k, nx, ny, nz)
    nxi = nx, ny, nz
    idx = (
        cart2lin((clamp(i - 1, 1, nx), clamp(j - 1, 1, ny), clamp(k - 1, 1, nz)), nxi),
        cart2lin((clamp(i, 1, nx), clamp(j - 1, 1, ny), clamp(k - 1, 1, nz)), nxi),
        cart2lin((clamp(i + 1, 1, nx), clamp(j - 1, 1, ny), clamp(k - 1, 1, nz)), nxi),
        cart2lin((clamp(i - 1, 1, nx), clamp(j, 1, ny), clamp(k - 1, 1, nz)), nxi),
        cart2lin((clamp(i - 1, 1, nx), clamp(j - 1, 1, ny), clamp(k, 1, nz)), nxi),
        cart2lin((clamp(i, 1, nx), clamp(j - 1, 1, ny), clamp(k, 1, nz)), nxi),
        cart2lin((clamp(i + 1, 1, nx), clamp(j - 1, 1, ny), clamp(k, 1, nz)), nxi),
        cart2lin((clamp(i - 1, 1, nx), clamp(j, 1, ny), clamp(k, 1, nz)), nxi),
        cart2lin((clamp(i - 1, 1, nx), clamp(j - 1, 1, ny), clamp(k + 1, 1, nz)), nxi),
        cart2lin((clamp(i, 1, nx), clamp(j - 1, 1, ny), clamp(k + 1, 1, nz)), nxi),
        cart2lin((clamp(i + 1, 1, nx), clamp(j - 1, 1, ny), clamp(k + 1, 1, nz)), nxi),
        cart2lin((clamp(i - 1, 1, nx), clamp(j, 1, ny), clamp(k + 1, 1, nz)), nxi),
    )
    return idx
end

function color_cells(nxi::NTuple{2,T}) where {T}
    color_list = ntuple(Val(2)) do i0
        [cart2lin((i, j), nxi) for j in i0:2:nxi[2] for i in i0:2:nxi[1]]
    end
    return color_list
end

function color_cells(nxi::NTuple{3,T}) where {T}
    color_list = ntuple(Val(3)) do i0
        [cart2lin((i, j), nxi) for j in i0:3:nxi[3] for j in i0:3:nxi[2] for i in i0:3:nxi[1]]
    end
    return color_list
end

@inline function cart2lin(I::NTuple{N,Integer}, nxi::NTuple{N,T}) where {N,T}
    return cart2lin(I..., ntuple(i -> nxi[i], Val(N - 1))...)
end
@inline cart2lin(i, j, nx) = i + (j - 1) * nx
@inline cart2lin(i, j, k, nx, ny) = cart2lin(i, j, nx) + (k - 1) * nx * ny

@inline corner_coordinate(grid, I::Integer) = grid[I]
@inline corner_coordinate(grid::NTuple{N,T1}, I::NTuple{N,T2}) where {T1,T2,N} = corner_coordinate(grid, I...)

@inline function corner_coordinate(grid::NTuple{N,T1}, I::Vararg{T2,N}) where {T1,T2,N}
    return ntuple(i -> grid[i][I[i]], Val(N))
end

@inline isincell(px::T, xv::T, dx::T) where {T<:Real} = xv < px < xv + dx 

@inline function isincell(p::NTuple{2,T}, xci::NTuple{2,T}, dxi::NTuple{2,T}) where {T}
    px, py = p # particle coordinate
    xc, yc = xci # corner coordinate
    dx, dy = dxi # spacing between gridpoints

    # check if it's outside the x-limits
    px < xc && return false
    px > xc + dx && return false
    # check if it's outside the y-limits
    py < yc && return false
    py > yc + dy && return false
    # otherwise particle is inside parent cell
    return true
end

@inline function isincell(p::NTuple{3,T}, xci::NTuple{3,T}, dxi::NTuple{3,T}) where {T}
    px, py, pz = p # particle coordinate
    xc, yc, zc = xci # corner coordinate
    dx, dy, dz = dxi # spacing between gridpoints

    # check if it's outside the x- and y-limits
    !isincell((px, py), (xc, yc), (dx, dy)) && return false
    # check if it's outside the z-limits
    pz < zc && return false
    pz > zc + dz && return false
    # otherwise particle is inside the cell
    return true
end

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


compute_dx(grid::LinRange{T, Int64}) where {T} = grid[2] - grid[1]

@inline function compute_dx(grid::NTuple{N,LinRange{T, Int64}}) where {N,T}
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

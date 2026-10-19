## ImplicitGlobalGrid support for MarkerChain
#
# Under a decomposed 2D global grid every rank holds the chain over its own columns,
# built from its local vertex coordinates; the ranks of a y-column hold identical
# copies. The collective operations below keep the copies consistent:
#
#   * `_update_chain_halo!`          — exchanges the x-halo of vertex-shaped chain data.
#   * `_reduce_chain_positions_y!`   — combines advected vertex positions across a
#     y-column so every vertex takes the value of the rank whose y-slab holds it.
#   * `_global_mean_height`          — mean height of the whole chain, each cell
#     counted once.

# `global_grid()` is ImplicitGlobalGrid's non-copying accessor (`get_global_grid()`
# would `deepcopy` the grid on every call).
@inline _distributed_chain() =
    ImplicitGlobalGrid.grid_is_initialized() && prod(ImplicitGlobalGrid.global_grid().dims) > 1

"""
    _update_chain_halo!(h)

Exchange the x-halo of the chain vertex vector `h` (one entry per local grid vertex)
between neighboring ranks. No-op when no global grid is initialized.
"""
function _update_chain_halo!(h::AbstractVector)
    ImplicitGlobalGrid.grid_is_initialized() || return nothing
    update_halo!(h; dims = (1,))
    return nothing
end

"""
    _owned_cells(nx)

Local cells, out of `nx`, that this rank counts along x: all of them except the overlap
cell on each side shared with another rank, so every global cell belongs to exactly
one rank of an x-row.
"""
function _owned_cells(nx::Integer)
    ImplicitGlobalGrid.grid_is_initialized() || return 1:nx
    gg = ImplicitGlobalGrid.global_grid()
    first = gg.coords[1] == 0 ? 1 : 2
    last = gg.coords[1] == gg.dims[1] - 1 ? nx : nx - 1
    return first:last
end

"""
    _global_mean_height(h, xv)

Width-weighted mean of the chain heights `h` at the vertices `xv`, over the whole
distributed chain. Equals [`mean_height`](@ref) when no global grid is initialized.
"""
function _global_mean_height(h, xv)
    _distributed_chain() || return mean_height(h, xv)
    cells = _owned_cells(length(h) - 1)
    widths = @views xv[cells .+ 1] .- xv[cells]
    local_sum = sum(@views (h[cells] .+ h[cells .+ 1]) .* widths)
    local_width = sum(widths)
    xrow = _column_comm(ImplicitGlobalGrid.global_grid().comm, (true, false, false))
    total_sum = MPI.Allreduce(local_sum, +, xrow)
    total_width = MPI.Allreduce(local_width, +, xrow)
    return total_sum / (2 * total_width)
end

"""
    _reduce_chain_positions_y!(px, py, h, yv)

Combine the advected chain vertex positions `(px, py)` across the y-column of a
decomposed global grid. `h` holds the vertex heights before advection and `yv` the
local y-coordinates of the vertices of the velocity grid.

Each rank only has the velocity of its own y-slab, so a vertex is owned by the ranks
whose slab, without the overlap cell on each side shared with another rank, holds its
starting height. An `Allreduce` over the y-column recovers the owners' positions by
weighted average; ranks sharing a vertex hold identical positions, so the average is
exact. Raises on every rank when a vertex lies outside every slab. No-op when no global
grid is initialized.
"""
function _reduce_chain_positions_y!(px, py, h, yv)
    ImplicitGlobalGrid.grid_is_initialized() || return nothing
    gg = ImplicitGlobalGrid.global_grid()
    y = Array(yv) # a short host copy: the slab bounds need its second and penultimate entries
    lo = gg.coords[2] > 0 ? y[2] : y[1]
    hi = gg.coords[2] < gg.dims[2] - 1 ? y[end - 1] : y[end]
    T = eltype(px)
    w = similar(px)
    @. w = ifelse((lo ≤ h) & (h ≤ hi), one(T), zero(T))

    if gg.dims[2] > 1
        # non-owners may hold non-finite extrapolations; zero them, not scale them
        @. px = ifelse(w > 0, px, zero(T))
        @. py = ifelse(w > 0, py, zero(T))
        ycol = _column_comm(gg.comm, (false, true, false))
        MPI.Allreduce!(px, +, ycol)
        MPI.Allreduce!(py, +, ycol)
        MPI.Allreduce!(w, +, ycol)
    end
    MPI.Allreduce(any(iszero, w), |, gg.comm) &&
        throw(ArgumentError("MarkerChain lies outside every local vertical velocity slab"))
    if gg.dims[2] > 1
        px ./= w
        py ./= w
    end
    return nothing
end

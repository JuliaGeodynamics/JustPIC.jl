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
    _check_chain_grid()
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
exact up to rounding. Raises on every rank when a vertex lies outside every slab, or
when an owned vertex is advected out of the local velocity grid across a rank
boundary. No-op when no global grid is initialized.
"""
function _reduce_chain_positions_y!(px, py, h, yv)
    ImplicitGlobalGrid.grid_is_initialized() || return nothing
    gg = ImplicitGlobalGrid.global_grid()
    below, above = gg.coords[2] > 0, gg.coords[2] < gg.dims[2] - 1
    # `yv` is increasing; reductions read its entries without scalar indexing
    ymin, ymax = minimum(yv), maximum(yv)
    lo = below ? minimum(@view yv[2:end]) : ymin
    hi = above ? maximum(@view yv[1:(end - 1)]) : ymax
    T = eltype(px)
    w = similar(px)
    @. w = ifelse((lo ≤ h) & (h ≤ hi), one(T), zero(T))
    # beyond the local velocity grid this rank has no velocity; checks the end point
    # of the trajectory only, not its intermediate stages
    too_far = mapreduce(|, w, py) do wi, yi
        (wi > 0) & ((below & (yi < ymin)) | (above & (yi > ymax)))
    end

    if gg.dims[2] > 1
        # non-owners may hold non-finite extrapolations; zero them, not scale them
        @. px = ifelse(w > 0, px, zero(T))
        @. py = ifelse(w > 0, py, zero(T))
        ycol = _column_comm(gg.comm, (false, true, false))
        MPI.Allreduce!(px, +, ycol)
        MPI.Allreduce!(py, +, ycol)
        MPI.Allreduce!(w, +, ycol)
    end
    outside, too_far = MPI.Allreduce([any(iszero, w), too_far], |, gg.comm)
    outside && throw(ArgumentError("MarkerChain lies outside every local vertical velocity slab"))
    too_far && error("a MarkerChain vertex moved more than one cell along y in one step; reduce dt")
    if gg.dims[2] > 1
        px ./= w
        py ./= w
    end
    return nothing
end

# One entry of a possibly device-resident vector, without scalar indexing.
_entry(a, i) = minimum(view(a, i:i))

"""
    _check_chain_remap(h_new, px, xv)

Raise when the remapped chain heights `h_new` are not valid, on every rank at once
under a decomposed global grid. `px` holds the advected x-positions of the local
vertices `xv`.

A distributed chain is remapped from the local vertices only, so the source of every
vertex this rank owns must lie between the advected local end vertices; it does
whenever no vertex moves more than one cell. Overlap vertices are refreshed by the
halo exchange and are not checked.
"""
function _check_chain_remap(h_new, px, xv)
    if !_distributed_chain()
        all(isfinite, h_new) || error(
            "Marker-chain remapping found non-finite trajectories or crossed vertices; reduce dt and check the velocity field"
        )
        return nothing
    end
    gg = ImplicitGlobalGrid.global_grid()
    cells = _owned_cells(length(h_new) - 1)
    owned = first(cells):(last(cells) + 1)
    too_far = (gg.coords[1] > 0 && _entry(xv, first(owned)) ≤ _entry(px, firstindex(px))) ||
        (gg.coords[1] < gg.dims[1] - 1 && _entry(xv, last(owned)) ≥ _entry(px, lastindex(px)))
    nonfinite = !all(isfinite, @view h_new[owned])
    too_far, nonfinite = MPI.Allreduce([too_far, nonfinite], |, gg.comm)
    too_far && error("a MarkerChain vertex moved more than one cell along x in one step; reduce dt")
    nonfinite && error(
        "Marker-chain remapping found non-finite trajectories or crossed vertices; reduce dt and check the velocity field"
    )
    return nothing
end

# Lagrangian chain markers would have to move between ranks, which is not implemented,
# and each rank only has the velocity of its own y-slab.
function _check_serial_chain_path(name)
    _distributed_chain() && throw(
        ArgumentError("$name does not support a decomposed global grid; use semilagrangian_advection_markerchain!")
    )
    return nothing
end

# The global grid is the same on every rank, so these throws need no reduction.
function _check_chain_grid()
    ImplicitGlobalGrid.grid_is_initialized() || return nothing
    gg = ImplicitGlobalGrid.global_grid()
    iszero(gg.periods[1]) ||
        throw(ArgumentError("a MarkerChain on a global grid does not support a periodic x direction"))
    # the ownership rules count one of the two shared cells on each side of a rank boundary
    all(d -> gg.dims[d] == 1 || gg.overlaps[d] == 2, 1:2) ||
        throw(ArgumentError("a MarkerChain on a global grid requires overlaps of 2 in every decomposed direction"))
    return nothing
end

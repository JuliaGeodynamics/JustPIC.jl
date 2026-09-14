## ImplicitGlobalGrid halo + z-column support for MarkerSurface
#
# The surface topography is a 2D vertex array (nx+1, ny+1) with no z dimension,
# so under a decomposed global grid it is replicated on every z-rank of a
# column. Two collective operations keep the replicas consistent and correct:
#
#   * `update_surface_halo!`        — exchanges the topography's x/y halos
#     (`update_halo!` restricted to dims (1,2); z is never touched).
#   * `reduce_surface_velocity_z!`  — under z-decomposition, combines the
#     interpolated surface velocities across the z-column so every node picks up
#     the value from the rank whose z-slab actually contains the surface.

"""
    update_surface_halo!(surf::MarkerSurface)

Exchange the x/y MPI halo of `surf.topo` between neighbouring ranks of the
active `ImplicitGlobalGrid` global grid. No-op when no global grid is
initialized (serial runs).

Called automatically at the end of `advect_surface_topo!`,
`smooth_surface_max_angle!`;
only needed explicitly after modifying `surf.topo` by hand.

# Notes
- `surf` must be built from the *local* (rank) vertex coordinates.
- Under MPI, use the global-grid periodicity (`periodx/periody` in
  `init_global_grid`) and leave `surf.periodic_1/periodic_2` as `false`;
  the local periodic flags wrap within the rank-local array.
"""
function update_surface_halo!(surf::MarkerSurface)
    _update_surface_halo!(surf.topo)
    return nothing
end

"""
    _update_surface_halo!(fields...)

Exchange the x/y `ImplicitGlobalGrid` halo of surface-shaped `fields`, which may
be nodal or cell-centered. No-op when no global grid is initialized.
"""
@inline function _update_surface_halo!(fields...)
    ImplicitGlobalGrid.grid_is_initialized() || return nothing
    # Only x/y are exchanged: surface fields have no z dimension.
    update_halo!(fields...; dims = (1, 2))
    return nothing
end

"""
    reduce_surface_velocity_z!(surf::MarkerSurface, zg)

Combine the interpolated surface velocities (`surf.vx/vy/vz`) across a z-column
of a decomposed global grid, and check that the surface lies within the vertical
extent of the velocity grid.

`zg` is the rank-local z vertex coordinate array/range of `Vz`. Its range is the
tightest of the three component grids, so a surface inside it also lies inside
the extended-center z grids of `Vx` and `Vy`.

Under z-decomposition each rank only interpolated the surface nodes inside its
own slab. Each rank marks a node as "owned" when `topo[i,j]` lies within its
local z-extent, contributes `(v·owned, owned)`, and an `Allreduce` over the
z-column recovers the value by weighted average. Nodes bracketed by two ranks
(shared overlap cells) hold identical velocities, so the average is exact.

Serial runs and grids with a single z-rank need no combining, since one slab
spans the whole vertical extent; the surface is still checked against it.

A node lying outside every slab has no interpolated velocity to recover and
raises — `_interp_vel_component` would otherwise clamp it to the boundary cell.
Every form of the check is reduced across the ranks so they raise together.
"""
function reduce_surface_velocity_z!(surf::MarkerSurface, zg)
    zlo, zhi = _z_extent(zg)
    local_outside = minimum(surf.topo) < zlo || maximum(surf.topo) > zhi
    if !ImplicitGlobalGrid.grid_is_initialized()
        local_outside &&
            throw(ArgumentError("MarkerSurface lies outside the vertical velocity grid"))
        return nothing
    end
    # `global_grid()` is IGG's non-copying internal accessor (`get_global_grid()`
    # would `deepcopy` the whole grid struct every call). Type-stable `::GlobalGrid`.
    igg = ImplicitGlobalGrid.global_grid()
    if igg.dims[3] == 1
        MPI.Allreduce(local_outside, |, igg.comm) &&
            throw(ArgumentError("MarkerSurface lies outside the vertical velocity grid on at least one rank"))
        return nothing
    end

    T = eltype(surf.topo)

    # Ownership weight, computed on-device: 1 where this rank's z-slab brackets the
    # surface node, 0 otherwise. Fused broadcast → one kernel, one array, no host copy.
    w = surf.z_ownership
    @. w = ifelse((zlo ≤ surf.topo) & (surf.topo ≤ zhi), one(T), zero(T))

    # Mask each velocity component in place (on device).
    surf.vx .*= w
    surf.vy .*= w
    surf.vz .*= w

    # Sum owned contributions across the z-column, straight on the device arrays —
    # same path ImplicitGlobalGrid uses for halos (GPU-aware MPI when enabled).
    zcomm = _z_column_comm(igg.comm)
    MPI.Allreduce!(surf.vx, +, zcomm)
    MPI.Allreduce!(surf.vy, +, zcomm)
    MPI.Allreduce!(surf.vz, +, zcomm)
    MPI.Allreduce!(w, +, zcomm)

    unowned = MPI.Allreduce(any(iszero, w), |, igg.comm)
    unowned && throw(ArgumentError("MarkerSurface lies outside every local vertical velocity slab"))

    # Weighted average → the owning rank's value (overlap ranks agree, so exact).
    surf.vx ./= w
    surf.vy ./= w
    surf.vz ./= w
    return nothing
end

"""
    _z_column_comm(comm)

Sub-communicator of `comm` joining the ranks that share an `(x,y)` column, i.e.
those differing only in their third Cartesian coordinate.

The Cartesian topology is fixed for the lifetime of a global grid, so the
communicator is built once and reused; `MPI.Cart_sub` would otherwise allocate
one per time step. The cache holds the parent communicator alongside it, which
both keeps that handle alive — so the identity test cannot match a recycled
handle value — and rebuilds the sub-communicator when a new global grid is
initialized. A dropped sub-communicator is freed by MPI.jl's finalizer; freeing
it here would be a collective call on ranks that may already have moved on.
"""
const Z_COLUMN_COMM = Ref{Tuple{MPI.Comm, MPI.Comm}}()

function _z_column_comm(comm::MPI.Comm)
    if isassigned(Z_COLUMN_COMM)
        cached_parent, zcomm = Z_COLUMN_COMM[]
        cached_parent == comm && return zcomm
    end
    zcomm = MPI.Cart_sub(comm, (false, false, true))
    Z_COLUMN_COMM[] = (comm, zcomm)
    return zcomm
end

# Local z-slab extent as scalars. `minimum`/`maximum` are GPU-friendly reductions
# (no scalar indexing, no host copy) and return the slab bounds for a monotonic
# z coordinate — cheap on the tiny (length nz+1) z line for ranges and arrays alike.
@inline _z_extent(zg) = (minimum(zg), maximum(zg))

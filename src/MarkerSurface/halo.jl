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
`smooth_surface_max_angle!` and each iteration of `smooth_surface_diffusive!`;
only needed explicitly after modifying `surf.topo` by hand.

# Notes
- `surf` must be built from the *local* (rank) vertex coordinates.
- Under MPI, use the global-grid periodicity (`periodx/periody` in
  `init_global_grid`) and leave `surf.periodic_1/periodic_2` as `false`;
  the local periodic flags wrap within the rank-local array.
"""
function update_surface_halo!(surf::MarkerSurface)
    ImplicitGlobalGrid.grid_is_initialized() || return nothing
    # Only x/y are exchanged: topo is a 2D field, its z "dimension" is trivial.
    update_halo!(surf.topo; dims = (1, 2))
    return nothing
end

"""
    reduce_surface_velocity_z!(surf::MarkerSurface, zg)

When the global grid is decomposed in z, combine the surface velocities
(`surf.vx/vy/vz`) across the ranks sharing this `(x,y)` column so that every
surface node `(i,j)` takes the trilinearly-interpolated velocity from whichever
rank's z-slab contains the surface elevation `topo[i,j]`. No-op for serial runs
or grids not decomposed in z.

`zg` is the rank-local z vertex coordinate array/range.

Implementation: each rank marks a node as "owned" when `topo[i,j]` lies within
its local z-extent, contributes `(v·owned, owned)`, and an `Allreduce` over the
z-column recovers the value by weighted average. Nodes bracketed by two ranks
(shared overlap cells) hold identical velocities, so the average is exact.
"""
function reduce_surface_velocity_z!(surf::MarkerSurface, zg)
    ImplicitGlobalGrid.grid_is_initialized() || return nothing
    # `global_grid()` is IGG's non-copying internal accessor (`get_global_grid()`
    # would `deepcopy` the whole grid struct every call). Type-stable `::GlobalGrid`.
    gg = ImplicitGlobalGrid.global_grid()
    gg.dims[3] > 1 || return nothing  # not decomposed in z → nothing to combine

    zlo, zhi = _z_extent(zg)  # local z-slab extent (scalars)
    T = eltype(surf.topo)

    # Ownership weight, computed on-device: 1 where this rank's z-slab brackets the
    # surface node, 0 otherwise. Fused broadcast → one kernel, one array, no host copy.
    w = similar(surf.topo)
    @. w = ifelse((zlo ≤ surf.topo) & (surf.topo ≤ zhi), one(T), zero(T))

    # Mask each velocity component in place (on device).
    surf.vx .*= w
    surf.vy .*= w
    surf.vz .*= w

    # Sum owned contributions across the z-column, straight on the device arrays —
    # same path ImplicitGlobalGrid uses for halos (GPU-aware MPI when enabled).
    zcomm = MPI.Cart_sub(gg.comm, (false, false, true))  # ranks sharing this (x,y) column
    MPI.Allreduce!(surf.vx, +, zcomm)
    MPI.Allreduce!(surf.vy, +, zcomm)
    MPI.Allreduce!(surf.vz, +, zcomm)
    MPI.Allreduce!(w, +, zcomm)
    MPI.free(zcomm)
    # ponytail: subcomm + weight array rebuilt per call; cache them in a
    # MarkerSurface workspace field for zero steady-state allocation if it profiles hot.

    # Weighted average → the owning rank's value (overlap ranks agree, so exact).
    # max(w,1) guards the physically-impossible node owned by no rank.
    surf.vx ./= max.(w, one(T))
    surf.vy ./= max.(w, one(T))
    surf.vz ./= max.(w, one(T))
    return nothing
end

# Local z-slab extent as scalars. `minimum`/`maximum` are GPU-friendly reductions
# (no scalar indexing, no host copy) and return the slab bounds for a monotonic
# z coordinate — cheap on the tiny (length nz+1) z line for ranges and arrays alike.
@inline _z_extent(zg) = (minimum(zg), maximum(zg))

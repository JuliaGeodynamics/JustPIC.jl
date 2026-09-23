"""
    move_particles!(particles::AbstractParticles, args; periodic_1=false, periodic_2=false, periodic_3=false)
    move_particles!(particles::AbstractParticles, grid, args, dxi; periodic_1=false, periodic_2=false, periodic_3=false)

Reassign particles to the correct parent cells after their coordinates have been
updated.

This routine keeps the coordinate arrays in `particles` and the companion fields
in `args` sorted by parent cell, preserving the package's spatially local memory
layout.

# Arguments
- `particles`: particle container whose coordinates have already been modified.
- `args`: tuple of per-particle fields that must move together with the particle
  coordinates.
- `grid`: optional vertex grid coordinates used by the lower-level method.
- `dxi`: optional grid spacing used by the lower-level method.
- `periodic_1`, `periodic_2`, `periodic_3`: enable periodic wrapping in the
  corresponding coordinate direction.

# Notes
- Particles that leave a non-periodic direction are discarded.
- Periodic directions use the ghost cells created by `add_periodic_ghost_nodes`
  to wrap coordinates and particle fields across opposite domain boundaries.
  The ghost cells of a periodic direction must be empty on entry, as they are
  after every call; otherwise an `ArgumentError` is thrown.
- A particle may cross any number of cells in one call, across periodic seams
  included. Jumps of more than one cell make the call slower: the cells are then
  transferred in `(2j₁ + 1) × … × (2jₙ + 1)` concurrent batches, with `jᵢ` the
  largest jump along direction `i`, so keep the displacement per step small.
- `args` must use the same cell layout as `particles.coords`.
- The public entry point uses the vertex grid and spacing stored in `particles`.
- If a destination cell is full, the particle is dropped and the number of
  dropped particles is returned. Companion fields are dropped with the particle.
"""
move_particles!(particles::AbstractParticles, args; periodic_1 = false, periodic_2 = false, periodic_3 = false) = move_particles!(particles, particles.xvi, args, particles.di.vertex; periodic_1 = periodic_1, periodic_2 = periodic_2, periodic_3 = periodic_3)

function move_particles!(particles::AbstractParticles, grid::NTuple{N}, args, dxi; periodic_1 = false, periodic_2 = false, periodic_3 = false) where {N}

    (; index) = particles
    N == 2 && periodic_3 && throw(ArgumentError("periodic_3 is only valid for 3D particles"))
    N in (2, 3) || throw(ArgumentError("The dimension of the problem must be either 2 or 3"))
    domain_limits = physical_domain_limits(particles)
    periodicity = ntuple(i -> (periodic_1, periodic_2, periodic_3)[i], Val(N))
    if any(periodicity)
        periodic_ghost_occupied(index, periodicity) && throw(
            ArgumentError("particles found in the ghost cells of a periodic direction; they must be empty before `move_particles!`")
        )
        wrap_particles!(particles, periodicity, domain_limits)
    end

    # The first sweep assumes that particles cross at most one cell along each direction, as
    # CFL-limited advection ensures. Particles that jump farther are left in place and moved by
    # a second sweep, which is sized to the largest jump along each direction.
    max_jump = ntuple(_ -> 1, Val(N))
    overflow = 0
    done, dropped = sweep_cells!(particles, grid, args, dxi, domain_limits, periodicity, max_jump)
    overflow += dropped
    while !done
        max_jump = maximum_particle_jump(particles, grid, dxi, domain_limits, periodicity)
        done, dropped = sweep_cells!(particles, grid, args, dxi, domain_limits, periodicity, max_jump)
        overflow += dropped
    end

    return overflow
end

function physical_domain_limits(particles::Particles{B, N}) where {B, N}
    return ntuple(i -> extrema(particles.xi_vel[i][i]), Val(N))
end

# Move the particles that are at most `max_jump[i]` cells away from their parent cell along each
# direction `i` and leave the others where they are. Return whether there were no others.
#
# The source cells are swept in colors: the cells of one color are processed concurrently, one
# color after the other. This is race free because a source cell only writes to itself and to
# the destination cells of its particles, which lie at most `max_jump[i]` cells away along
# direction `i` (cyclically along periodic directions, where the physical cells `2:n - 1` form a
# ring). Two cells of the same color are more than `2 * max_jump[i]` cells apart along at least
# one direction `i`, so their write sets are disjoint and no two threads ever search or fill
# slots in the same cell.
function sweep_cells!(particles, grid, args, dxi, domain_limits, periodicity, max_jump)
    (; coords, index) = particles
    backend = ka_backend(index)
    deferred = KernelAbstractions.zeros(backend, Int, 1)
    overflow = KernelAbstractions.zeros(backend, Int, 1)
    bound = (; max_jump, periodicity, deferred, overflow)
    layout = map(ColorAxis, size(index), max_jump, periodicity)
    nblocks = map(axis -> axis.nblocks, layout)
    for colors in Iterators.product(map(axis -> 1:axis.ncolors, layout)...)
        launch!(
            backend, move_particles_ps!, nblocks,
            coords, grid, dxi, index, domain_limits, args, bound, colors, layout
        )
    end
    return iszero(maximum(deferred)), maximum(overflow)
end

# Partition of the cells of one direction into colors. The cell of color `k` in block `b` is
# given by `color_cell`, and cells of the same color in consecutive blocks are at least
# `2 * max_jump + 1` cells apart. Along a periodic direction the physical cells `2:n - 1` form
# a ring of `period = n - 2` cells and the blocks tile it, so the last block is also separated
# from the first one across the seam; the ghost cells are not scheduled.
struct ColorAxis
    ncolors::Int
    nblocks::Int
    ncells::Int
    period::Int # 0 unless periodic
end

function ColorAxis(n::Integer, max_jump::Integer, periodic::Bool)
    separation = 2 * max_jump + 1
    if periodic
        period = n - 2
        nblocks = max(period ÷ separation, 1)
        return ColorAxis(cld(period, nblocks), nblocks, n, period)
    end
    ncolors = min(separation, n)
    return ColorAxis(ncolors, cld(n, ncolors), n, 0)
end

# Cell index of color `k` in block `b`, or 0 if that color has no cell in the block.
@inline function color_cell(axis::ColorAxis, k::Integer, b::Integer)
    if iszero(axis.period)
        i = axis.ncolors * (b - 1) + k
        return ifelse(i ≤ axis.ncells, i, 0)
    end
    first = ((b - 1) * axis.period + axis.nblocks - 1) ÷ axis.nblocks
    next = (b * axis.period + axis.nblocks - 1) ÷ axis.nblocks
    return ifelse(k ≤ next - first, 2 + first + k - 1, 0)
end

@kernel function move_particles_ps!(
        coords, grid, dxi, index, domain_limits, args, bound, colors::NTuple{N}, layout::NTuple{N}
    ) where {N}
    I = @index(Global, NTuple)
    indices = ntuple(i -> color_cell(layout[i], colors[i], I[i]), Val(N))

    if all(>(0), indices)
        _move_particles!(coords, grid, dxi, index, domain_limits, indices, args, bound)
    end
end

# Largest number of cells any particle has to cross along each direction to reach its parent
# cell. Particles that are outside the domain or already in their parent cell do not count.
function maximum_particle_jump(particles, grid, dxi, domain_limits, periodicity)
    (; coords, index) = particles
    backend = ka_backend(index)
    max_jumps = map(_ -> KernelAbstractions.zeros(backend, Int, size(index)...), periodicity)
    launch!(
        backend, maximum_particle_jump!, size(index),
        max_jumps, coords, grid, dxi, index, domain_limits, periodicity
    )
    return map(maximum, max_jumps)
end

@kernel function maximum_particle_jump!(
        max_jumps, coords, grid, di, index, domain_limits, periodicity
    )
    I = @index(Global, NTuple)
    corner_xi = corner_coordinate(grid, I)
    dxi = @dxi di I...

    max_jump = map(_ -> 0, periodicity)
    for ip in cellaxes(index)
        doskip(index, ip, I...) && continue
        pᵢ = cache_particle(coords, ip, I)
        isincell(pᵢ, corner_xi, dxi) && continue
        indomain(pᵢ, domain_limits) || continue
        new_cell = find_parent_cell_bisection(pᵢ, grid, I)
        max_jump = max.(max_jump, cell_jump(new_cell, I, size(index), periodicity))
    end

    for d in eachindex(max_jumps)
        max_jumps[d][I...] = max_jump[d]
    end
end

# The ghost cells of a periodic direction are not swept, so they have to be empty.
function periodic_ghost_occupied(index, periodicity)
    backend = ka_backend(index)
    occupied = KernelAbstractions.zeros(backend, Int, 1)
    launch!(backend, periodic_ghost_occupied!, size(index), occupied, index, periodicity)
    return !iszero(maximum(occupied))
end

@kernel function periodic_ghost_occupied!(occupied, index, periodicity)
    I = @index(Global, NTuple)
    if in_periodic_ghost(I, size(index), periodicity)
        for ip in cellaxes(index)
            doskip(index, ip, I...) && continue
            occupied[1] = 1
        end
    end
end

@inline function in_periodic_ghost(I::NTuple{N}, nxi::NTuple{N}, periodicity::NTuple{N}) where {N}
    return any(ntuple(i -> periodicity[i] & (I[i] == 1 || I[i] == nxi[i]), Val(N)))
end

@inline function axis_jump(dst, src, n, periodic)
    jump = abs(dst - src)
    return ifelse(periodic, min(jump, n - 2 - jump), jump)
end

@inline function cell_jump(dst::NTuple{N}, src::NTuple{N}, nxi::NTuple{N}, periodicity::NTuple{N}) where {N}
    return ntuple(i -> axis_jump(dst[i], src[i], nxi[i], periodicity[i]), Val(N))
end

function _move_particles!(coords, grid, dxi, index, domain_limits, idx, args, bound)
    # coordinate of the lower-most-left coordinate of the parent cell
    corner_xi = corner_coordinate(grid, idx)
    # iterate over neighbouring (child) cells
    move_kernel!(coords, corner_xi, grid, dxi, index, domain_limits, args, idx, bound)
    return nothing
end

function move_kernel!(
        coords,
        corner_xi,
        grid,
        di,
        index,
        domain_limits,
        args::NTuple{N2, T},
        idx::NTuple{N1, Int64},
        bound,
    ) where {N1, N2, T}

    dxi = @dxi di idx...

    # iterate over particles in child cell
    for ip in cellaxes(index)
        doskip(index, ip, idx...) && continue
        pᵢ = cache_particle(coords, ip, idx)

        # check whether the particle is
        # within the same cell and skip it
        isincell(pᵢ, corner_xi, dxi) && continue

        # particle went of of the domain, get rid of it
        domain_check = !(indomain(pᵢ, domain_limits))
        if domain_check
            CAI.@index index[ip, idx...] = false
            empty_particle!(coords, ip, idx)
            empty_particle!(args, ip, idx)
        end
        domain_check && continue

        new_cell = find_parent_cell_bisection(pᵢ, grid, idx)

        # too far for the cells swept concurrently: leave it for a wider sweep
        if any(cell_jump(new_cell, idx, size(index), bound.periodicity) .> bound.max_jump)
            bound.deferred[1] = 1
            continue
        end

        # hold particle variables
        current_args = cache_args(args, ip, idx)

        # remove particle from child cell
        CAI.@index index[ip, idx...] = false
        empty_particle!(coords, ip, idx)
        empty_particle!(args, ip, idx)

        # check whether there's empty space in parent cell
        free_idx = find_free_memory(index, new_cell...)
        if iszero(free_idx)
            KernelAbstractions.@atomic bound.overflow[1] += 1
            continue
        end

        # move particle and its fields to the first free memory location
        CAI.@index index[free_idx, new_cell...] = true
        fill_particle!(coords, pᵢ, free_idx, new_cell)
        fill_particle!(args, current_args, free_idx, new_cell)
    end
    return nothing
end

@generated function look_around(px::NTuple{N, Number}, x::NTuple{N, AbstractVector}, I) where {N}
    return quote
        @inline
        Base.@ntuple $N i -> look_around(px[i], x[i], I[i])
    end
end

function look_around(px, x, I)
    for i in -1:1
        ii = I + i
        ii = clamp(ii, 1, length(x))
        x[ii] ≤ px ≤ x[ii + 1] && return clamp(I + i, 1, length(x))
        # !(1 ≤ ii ≤ length(x)) && continue
        # x[ii] ≤ px ≤ x[ii + 1] && return I + i
    end
    return Inf
end

## Utility functions

function cell_index_neighbour(
        xᵢ::NTuple{N}, xcᵢ::NTuple{N}, dxᵢ::NTuple{N}, I::NTuple{N, Integer}
    ) where {N}
    return ntuple(Val(N)) do i
        cell_index_neighbour(xᵢ[i], xcᵢ[i], dxᵢ[i], I[i], I)
    end
end

# Case: regular grid
function cell_index_neighbour(x, xc, dx::Number, i::Integer, I)
    xR = xc + dx
    (xc ≤ x ≤ xR) && return i
    (xc - dx < x < xc) && return i - 1
    (xR < x < xc + 2 * dx) && return i + 1
    return error("Particle moved more than one cell away from the parent cell $I")
end

# Case: regularly refined grid
function cell_index_neighbour(x, xC, dx::AbstractVector, i::Integer, I)
    n = length(dx)
    isleftboundary = i == 1
    isrightboundary = i == n
    # grid sizes
    dxL = dx[i - 1 * !isleftboundary]  # left cell
    dxC = dx[i]                        # center cell
    dxR = dx[i + 1 * !isrightboundary] # right cell
    # grid corners
    xL = xC - dxL
    xR1 = xC + dxC
    xR2 = xR1 + dxR
    # check where the particle is
    (xL < x < xC)   && return i - 1
    (xC ≤ x ≤ xR1)  && return i
    (xR1 < x < xR2) && return i + 1

    return error("Particle moved more than one cell away from the parent cell $I in $i, with xi $x")
end

function find_free_memory(index, I::Vararg{Int, N}) where {N}
    for i in cellaxes(index)
        (CAI.@index(index[i, I...])) || return i
    end
    return 0
end

# half-open `[xmin, xmax)`, matching `isincell` and the interval `wrap_coordinate` maps into
@generated function indomain(p::NTuple{N, T1}, domain_limits::NTuple{N, T2}) where {N, T1, T2}
    return quote
        Base.@_inline_meta
        Base.Cartesian.@nexprs $N i ->
        ((domain_limits[i][1] ≤ p[i] < domain_limits[i][2]) || return false)
        return true
    end
end

@generated function indomain(idx_child::NTuple{N, Integer}, nxi::NTuple{N, Integer}) where {N}
    return quote
        Base.@_inline_meta
        Base.Cartesian.@nexprs $N i ->
        (1 ≤ idx_child[i] ≤ nxi[i] - 1) == false && return false
        return true
    end
end

@generated function isparticleempty(p::NTuple{N, T}) where {N, T}
    return quote
        Base.@_inline_meta
        Base.Cartesian.@nexprs $N i -> isnan(p[i]) && return true
        return false
    end
end

@inline function cache_args(args::NTuple{N1, T}, ip, I::NTuple{N2, Int64}) where {T, N1, N2}
    return ntuple(i -> (CAI.@index(args[i][ip, I...])), Val(N1))
end

@inline function cache_args(args::NTuple{N}, ip, I::Integer) where {N}
    return ntuple(i -> (CAI.@index(args[i][ip, I])), Val(N))
end

@inline function cache_particle(
        p::NTuple{N1, T}, ip, I::Union{Integer, NTuple{N2, Integer}}
    ) where {T, N1, N2}
    return cache_args(p, ip, I)
end

@inline function child_index(parent_cell::NTuple{N, Int64}, I::NTuple{N, Int64}) where {N}
    return ntuple(i -> parent_cell[i] + I[i], Val(N))
end

@generated function empty_particle!(
        p::NTuple{N1, T}, ip, I::NTuple{N2, Int64}
    ) where {N1, N2, T}
    return quote
        Base.@_inline_meta
        Base.Cartesian.@nexprs $N1 i ->
        CAI.@index p[i][ip, I...] = convert(eltype(eltype(p[i])), NaN)
    end
end

@generated function empty_particle!(p::NTuple{N}, ip, I::Integer) where {N}
    return quote
        Base.@_inline_meta
        Base.Cartesian.@nexprs $N i ->
        CAI.@index p[i][ip, I] = convert(eltype(eltype(p[i])), NaN)
    end
end

@inline function fill_particle!(
        p::NTuple{N, T1}, field::NTuple{N, T2}, ip, I::Int64
    ) where {N, T1, T2}
    return fill_particle!(p, field, ip, (I,))
end

@generated function fill_particle!(
        p::NTuple{N1, T1}, field::NTuple{N1, T2}, ip, I::NTuple{N2, Int64}
    ) where {N1, N2, T1, T2}
    return quote
        Base.Cartesian.@nexprs $N1 i -> begin
            Base.@_inline_meta
            tmp = p[i]
            CAI.@index tmp[ip, I...] = field[i]
        end
        return nothing
    end
end

"""
    clean_particles!(particles, grid, args)

Remove invalid or inactive particle slots and keep particle-associated fields in
`args` consistent with the particle storage layout.

This is typically used after particle deletion or reinjection to compact each
cell's active particle block.
"""
function clean_particles!(particles::Particles, grid, args)
    (; coords, index) = particles
    dxi = compute_dx(grid)
    ni = size(index)
    launch!(ka_backend(index), _clean!, ni, coords, grid, dxi, index, args)
    return nothing
end

@kernel function _clean!(particle_coords, grid, dxi, index, args)
    I = @index(Global, NTuple)
    clean_kernel!(particle_coords, grid, @dxi(dxi, I...), index, args, I...)
end

function clean_kernel!(
        particle_coords, grid, dxi, index, args, cell_indices::Vararg{Int, N}
    ) where {N}
    corner_xi = corner_coordinate(grid, cell_indices...)
    # iterate over particles in child cell
    for ip in cellaxes(index)
        pᵢ = cache_particle(particle_coords, ip, cell_indices)

        if CAI.@index index[ip, cell_indices...] # true if memory allocation is filled with a particle
            if !(isincell(pᵢ, corner_xi, dxi))
                # remove particle from child cell
                CAI.@index index[ip, cell_indices...] = false
                empty_particle!(particle_coords, ip, cell_indices)
                empty_particle!(args, ip, cell_indices)
            end
        end
    end
    return nothing
end

function global_domain_limits(origin::NTuple{N, Any}, dxi::NTuple{N, Any}) where {N}
    fn = nx_g, ny_g, nz_g

    lims = ntuple(Val(N)) do i
        Base.@_inline_meta
        origin[i], (fn[i]() - 1) * dxi[i]
    end

    return lims
end

# The following kernels are used in the `move_particles!` function
# to remove a random particle from the memory location so that the
# cell capacity is always below 80% of its maximum.
@kernel function empty_particles!(coords, index, cell_length, args)
    I = @index(Global, NTuple)
    empty_kernel!(coords, index, cell_length, args, I)
end

function empty_kernel!(
        coords, index, cell_length, args::NTuple{N2}, I::NTuple{N1, Int64}
    ) where {N1, N2}

    # count number of active particles inside I-th cell
    number_of_particles = count_particles(index, I...)
    # if the number of particles is less than 75% of the cell length then we do
    # nothing; integer arithmetic (Float64 in kernels breaks Metal)
    max_particles_allowed = (3 * cell_length) ÷ 4
    number_of_particles < max_particles_allowed && return nothing

    # else we randomly remove particles until we are below 75% capacity
    number_of_particles_to_remove = number_of_particles - max_particles_allowed
    counter = 0
    while counter < number_of_particles_to_remove
        # randomly select a particle to remove
        index_to_remove = rand(1:number_of_particles)
        # check if a particle is actually in that memory location
        doskip(index, index_to_remove, I...) && continue
        # great, lets get rid of it
        CAI.@index index[index_to_remove, I...] = false
        empty_particle!(coords, index_to_remove, I)
        empty_particle!(args, index_to_remove, I)
        counter += 1
    end
    return nothing
end

function count_particles(index, I::Vararg{Int, N}) where {N}
    count = 0
    for i in cellaxes(index)
        count += CAI.@index index[i, I...]
    end
    return count
end


######

function wrap_particles!(particles, periodicity, domain_limits)
    (; index, coords) = particles
    ni = size(index)
    launch!(ka_backend(index), wrap_particles_kernel!, ni, index, coords, periodicity, domain_limits)
    return nothing
end

@kernel function wrap_particles_kernel!(index, coords, periodicity, domain_limits)
    I = @index(Global, NTuple)
    for ip in cellaxes(index)
        doskip(index, ip, I...) && continue
        for dim in eachindex(coords)
            if periodicity[dim]
                px = CAI.@index coords[dim][ip, I...]
                CAI.@index coords[dim][ip, I...] = wrap_coordinate(
                    px, periodicity[dim], domain_limits[dim]
                )
            end
        end
    end
end

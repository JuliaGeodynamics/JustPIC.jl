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
- `args` must use the same cell layout as `particles.coords`.
- The public entry point uses the vertex grid and spacing stored in `particles`.
"""
move_particles!(particles::AbstractParticles, args; periodic_1 = false, periodic_2 = false, periodic_3 = false) = move_particles!(particles, particles.xvi, args, particles.di.vertex; periodic_1 = periodic_1, periodic_2 = periodic_2, periodic_3 = periodic_3)

function move_particles!(particles::AbstractParticles, grid::NTuple{N}, args, dxi; periodic_1 = false, periodic_2 = false, periodic_3 = false) where {N}

    (; coords, index, max_xcell) = particles
    nxi = size(index)
    domain_limits = extrema.(grid)
    n_color = ntuple(i -> ceil(Int, nxi[i] / 3), Val(N))
    periodicity = periodic_1, periodic_2, periodic_3 
    isperiodic = any(periodicity)
    
    # first of all, we need to empty ghost nodes to make sure that no particles are moved into them
    # if !isperiodic    
    #     empty_ghost_nodes!(particles, args)
    # end

    if any(periodicity)
        wrap_fields!(particles, periodicity, args)
    end

    # make some space for incoming particles
    # @parallel (@idx nxi) empty_particles!(coords, index, max_xcell, args)
    # move particles
    if N == 2 # 2D case
        nthreads = (16, 16)
        nblocks = ceil.(Int, n_color ./ nthreads)
        for offsetᵢ in 1:3, offsetⱼ in 1:3
            @parallel (@idx n_color) nblocks nthreads move_particles_ps!(
                coords, grid, dxi, index, domain_limits, args, (offsetᵢ, offsetⱼ)
            )
        end
    elseif N == 3 # 3D case
        nthreads = (16, 16, 1)
        nblocks = ceil.(Int, n_color ./ nthreads)
        for offsetᵢ in 1:3, offsetⱼ in 1:3, offsetₖ in 1:3
            @parallel (@idx n_color) nblocks nthreads move_particles_ps!(
                coords, grid, dxi, index, domain_limits, args, (offsetᵢ, offsetⱼ, offsetₖ)
            )
        end
    else
        error(ThrowArgument("The dimension of the problem must be either 2 or 3"))
    end

    return nothing
end

@parallel_indices (I...) function move_particles_ps!(
        coords, grid, dxi, index, domain_limits, args, offsets::NTuple{N}
    ) where {N}
    indices = ntuple(Val(N)) do i
        3 * (I[i] - 1) + offsets[i]
    end

    if all(indices .≤ size(index))
        _move_particles!(coords, grid, dxi, index, domain_limits, indices, args)
    end
    return nothing
end

function _move_particles!(coords, grid, dxi, index, domain_limits, idx, args)
    # coordinate of the lower-most-left coordinate of the parent cell
    corner_xi = corner_coordinate(grid, idx)
    # iterate over neighbouring (child) cells
    move_kernel!(coords, corner_xi, grid, dxi, index, domain_limits, args, idx)
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
    ) where {N1, N2, T}

    starting_point = 1
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
            @index index[ip, idx...] = false
            empty_particle!(coords, ip, idx)
            empty_particle!(args, ip, idx)
        end
        domain_check && continue

        new_cell = find_parent_cell_bisection(pᵢ, grid, idx)

        # hold particle variables
        current_args = cache_args(args, ip, idx)

        # remove particle from child cell
        @index index[ip, idx...] = false
        empty_particle!(coords, ip, idx)
        empty_particle!(args, ip, idx)

        # check whether there's empty space in parent cell
        free_idx = find_free_memory(starting_point, index, new_cell...)
        iszero(free_idx) && continue
        starting_point = free_idx

        # move particle and its fields to the first free memory location
        @index index[free_idx, new_cell...] = true
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
        (@index(index[i, I...])) || return i
    end
    return 0
end

function find_free_memory(initial_index::Integer, index::CellArray, I::Vararg{Int, N}) where {N}
    for i in initial_index:cellnum(index)
        (@index(index[i, I...])) || return i
    end
    return 0
end

@generated function indomain(p::NTuple{N, T1}, domain_limits::NTuple{N, T2}) where {N, T1, T2}
    return quote
        Base.@_inline_meta
        Base.Cartesian.@nexprs $N i ->
        ((domain_limits[i][1] < p[i] < domain_limits[i][2]) || return false)
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
    return ntuple(i -> (@index(args[i][ip, I...])), Val(N1))
end

@inline function cache_args(args::NTuple{N}, ip, I::Integer) where {N}
    return ntuple(i -> (@index(args[i][ip, I])), Val(N))
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
        Base.Cartesian.@nexprs $N1 i -> @index p[i][ip, I...] = NaN
    end
end

@generated function empty_particle!(p::NTuple{N}, ip, I::Integer) where {N}
    return quote
        Base.@_inline_meta
        Base.Cartesian.@nexprs $N i -> @index p[i][ip, I] = NaN
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
            @index tmp[ip, I...] = field[i]
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
    @parallel (@idx ni) _clean!(coords, grid, dxi, index, args)
    return nothing
end

@parallel_indices (i, j) function _clean!(
        particle_coords::NTuple{2, Any}, grid::NTuple{2, Any}, dxi::NTuple{2, Any}, index, args
    )
    clean_kernel!(particle_coords, grid, @dxi(dxi, i, j), index, args, i, j)
    return nothing
end

@parallel_indices (i, j, k) function _clean!(
        particle_coords::NTuple{3, Any}, grid::NTuple{3, Any}, dxi::NTuple{3, Any}, index, args
    )
    clean_kernel!(particle_coords, grid, @dxi(dxi, i, j, k), index, args, i, j, k)
    return nothing
end

function clean_kernel!(
        particle_coords, grid, dxi, index, args, cell_indices::Vararg{Int, N}
    ) where {N}
    corner_xi = corner_coordinate(grid, cell_indices...)
    # iterate over particles in child cell
    for ip in cellaxes(index)
        pᵢ = cache_particle(particle_coords, ip, cell_indices)

        if @index index[ip, cell_indices...] # true if memory allocation is filled with a particle
            if !(isincell(pᵢ, corner_xi, dxi))
                # remove particle from child cell
                @index index[ip, cell_indices...] = false
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
@parallel_indices (I...) function empty_particles!(coords, index, cell_length, args)
    empty_kernel!(coords, index, cell_length, args, I)
    return nothing
end

function empty_kernel!(
        coords, index, cell_length, args::NTuple{N2}, I::NTuple{N1, Int64}
    ) where {N1, N2}

    # count number of active particles inside I-th cell
    number_of_particles = count_particles(index, I...)
    # if the number of particles is less than 80%
    # of the cell length then we do nothing
    max_particles_allowed = cell_length * 0.75
    number_of_particles < max_particles_allowed && return nothing

    # else we randomly remove particles until we are below 80% capacity
    number_of_particles_to_remove =
        number_of_particles - round(Int, max_particles_allowed, RoundDown)
    counter = 0
    while counter < number_of_particles_to_remove
        # randomly select a particle to remove
        index_to_remove = rand(1:number_of_particles)
        # check if a particle is actually in that memory location
        doskip(index, index_to_remove, I...) && continue
        # great, lets get rid of it
        @index index[index_to_remove, I...] = false
        empty_particle!(coords, index_to_remove, I)
        empty_particle!(args, index_to_remove, I)
        counter += 1
    end
    return nothing
end

function count_particles(index, I::Vararg{Int, N}) where {N}
    count = 0
    for i in cellaxes(index)
        count += @index index[i, I...]
    end
    return count
end


###### 

empty_ghost_nodes!(particles, others::NTuple{N, Any}) where N = empty_ghost_nodes!(particles, others...)

function empty_ghost_nodes!(particles, others...)

    (; index, coords) = particles
    ni = size(index)
    @parallel (@idx ni) empty_ghost_nodes!(index, (coords..., others...))
    
    return nothing
end

@parallel_indices (I...) function empty_ghost_nodes!(index, others)

    if isghost(size(index), I)
        @inbounds for ip in cellaxes(index)
            @inbounds @index index[ip, I...] = false
            for other in others
                @index other[ip, I...] = NaN
            end
        end
    end

    return nothing
end

@generated function isghost(sz::NTuple{N, Int}, I::NTuple{N}) where {N}
    return quote
        @inline
        Base.@nany $N i -> @inbounds isequal(sz[i], I[i]) || @inbounds isequal(1, I[i])
    end
end


###### 

wrap_fields!(particles, periodicity, others::NTuple{N, Any}) where N = wrap_fields!(particles, periodicity, others...)

function wrap_fields!(particles, periodicity, others...)

    (; index, coords, xvi) = particles
    ni = size(index)

    @parallel (@idx ni) wrap_fields!(index, coords, (others...,), xvi, periodicity)

    return nothing
end

@parallel_indices (I...) function wrap_fields!(index, coords, others, xvi, periodicity)

    if isghost(size(index), I)
     
        I_wrapped = wrap_index(periodicity, size(index), I)

        @inbounds for ip in cellaxes(index)
            # @index index[ip, I_wrapped...] = @index(index[ip, I...])
            @index index[ip, I...] = @index(index[ip, I_wrapped...])
            wrap_coordinates!(periodicity, coords, xvi, ip, I_wrapped, I) 
            for other in others
                # @index other[ip, I_wrapped...] = @index(other[ip, I...])
                @index other[ip, I...] = @index(other[ip, I_wrapped...])
            end
        end
    end

    return nothing
end

@generated function wrap_coordinates!(periodicity, coords::NTuple{N}, xvi, ip, I_wrapped, I::NTuple{N}) where {N}
    return quote
        @inline
        Base.@nexprs $N i -> begin
            coordsᵢ = coords[i]
            px      = @index coordsᵢ[ip, I_wrapped...]
            xmax    = xvi[i][end-1]
            xmin    = xvi[i][2]
            if periodicity[i]
                if px > (xmax + xmin) / 2
                    Δx = xmax - px
                    px_new = xmin - Δx
                    @index coordsᵢ[ip, I...] = px_new
                else
                    Δx = px - xmin
                    px_new = xmax + Δx
                    @index coordsᵢ[ip, I...] = px_new
                end
            else
                @index coordsᵢ[ip, I...] = px
            end
        end
    end
end

@generated function wrap_index(periodicity, sz::NTuple{N, Int}, I::NTuple{N}) where {N}
    return quote
        @inline
        Base.@ntuple $N i -> begin
            if periodicity[i]
                Iᵢ = wrap_index(I[i], sz[i])
            else
                I[i]
            end
        end
    end
end

@inline function wrap_index(i::Integer, idx_max::Integer)
    i == 1 && return idx_max-1
    i == idx_max && return 1
    return i
end

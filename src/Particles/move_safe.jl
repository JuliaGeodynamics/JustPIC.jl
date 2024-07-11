"""
    move_particles!(particles::AbstractParticles, grid, args)

Move particles in the given `particles` container according to the provided `grid` and particles fields in `args`.

# Arguments
- `particles`: The container of particles to be moved.
- `grid`: The grid used for particle movement.
- `args`: `CellArrays`s containing particle fields.
"""
function move_particles!(particles::AbstractParticles, grid, args)
    # implementation goes here
    dxi                          = compute_dx(grid)
    (; coords, index, max_xcell) = particles
    nxi                          = size(index)
    domain_limits                = extrema.(grid)

    # make some space for incoming particles
    @parallel (@idx nxi) empty_particles!(coords, index, max_xcell, args)
    # move particles 
    n_color = ntuple(Val(length(nxi))) do i
        ceil(Int, nxi[i] * 0.5)
    end
    
    if length(nxi) == 2
        for offset_x in 1:2, offset_y in 1:2
            @parallel (@idx n_color) move_particles_ps!(coords, grid, dxi, index, domain_limits, args, offset_x, offset_y)
        end
    
    elseif length(nxi) == 3
        # for offset_x in 1:2, offset_y in 1:2, offset_z in 1:2
        #     @parallel (@idx n_color) move_particles_ps!(coords, grid, dxi, index, domain_limits, args, offset_x, offset_y, offset_z)
        # end
    
    end

    return nothing
end

@parallel_indices (I...) function move_particles_ps!(
    coords, grid, dxi, index, domain_limits, args, offset_x, offset_y
)
    nx, ny = size(index)
    i      = 2 * (I[1] - 1) + offset_x
    j      = 2 * (I[2] - 1) + offset_y

    if (i < nx) && (j < ny)
        _move_particles!(coords, grid, dxi, index, domain_limits, (i,j), args)
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
    dxi,
    index,
    domain_limits,
    args::NTuple{N2,T},
    idx::NTuple{N1,Int64},
) where {N1,N2,T}

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
            println("Particle $(pᵢ) went out of the domain")
            @cell index[ip, idx...] = false
            empty_particle!(coords, ip, idx)
            empty_particle!(args, ip, idx)
        end
        domain_check && continue

        # new cell indices
        # new_cell = ntuple(Val(N1)) do i
        #     cell_index(pᵢ[i], grid[i], dxi[i])
        # end

        new_cell = cell_index_neighbour(pᵢ, corner_xi, dxi, idx)

        # hold particle variables
        current_args = @inbounds cache_args(args, ip, idx)
        # remove particle from child cell
        @inbounds @cell index[ip, idx...] = false
        empty_particle!(coords, ip, idx)
        empty_particle!(args, ip, idx)
        # check whether there's empty space in parent cell
        free_idx = find_free_memory(index, new_cell...)
        free_idx == 0 && @show free_idx
        free_idx == 0 && continue
        # move particle and its fields to the first free memory location
        @inbounds @cell index[free_idx, new_cell...] = true
        fill_particle!(coords, pᵢ, free_idx, new_cell)
        fill_particle!(args, current_args, free_idx, new_cell)
    end
end

## Utility functions

function cell_index_neighbour(x, xc, dx, i::Integer)
    xR = xc + dx
    (xc ≤ x ≤ xR)          && return i 
    (xc - dx < x < xc)     && return i - 1
    (xR < x < xc + 2 * dx) && return i + 1
    error("Particle moved more than one cell away from the parent cell")
end

function cell_index_neighbour(xᵢ::NTuple{N}, xcᵢ::NTuple{N}, dxᵢ::NTuple{N}, I::NTuple{N,Integer}) where N
    ntuple(Val(N)) do i 
        cell_index_neighbour(xᵢ[i], xcᵢ[i], dxᵢ[i], I[i])
    end
end

function find_free_memory(index, I::Vararg{Int,N}) where {N}
    for i in cellaxes(index)
        !(@cell(index[i, I...])) && return i
    end
    return 0
end

@generated function indomain(p::NTuple{N,T1}, domain_limits::NTuple{N,T2}) where {N,T1,T2}
    quote
        Base.@_inline_meta
        Base.Cartesian.@nexprs $N i ->
            (@inbounds !(domain_limits[i][1] < p[i] < domain_limits[i][2]) && return false)
        return true
    end
end

@generated function indomain(idx_child::NTuple{N,Integer}, nxi::NTuple{N,Integer}) where {N}
    quote
        Base.@_inline_meta
        Base.Cartesian.@nexprs $N i ->
            @inbounds (1 ≤ idx_child[i] ≤ nxi[i] - 1) == false && return false
        return true
    end
end

@generated function isparticleempty(p::NTuple{N,T}) where {N,T}
    quote
        Base.@_inline_meta
        Base.Cartesian.@nexprs $N i -> @inbounds isnan(p[i]) && return true
        return false
    end
end

@inline function cache_args(args::NTuple{N1,T}, ip, I::NTuple{N2,Int64}) where {T,N1,N2}
    return ntuple(i -> @cell(args[i][ip, I...]), Val(N1))
end

@inline function cache_particle(p::NTuple{N1,T}, ip, I::NTuple{N2,Int64}) where {T,N1,N2}
    return cache_args(p, ip, I)
end

@inline function child_index(parent_cell::NTuple{N,Int64}, I::NTuple{N,Int64}) where {N}
    return ntuple(i -> parent_cell[i] + I[i], Val(N))
end

@generated function empty_particle!(
    p::NTuple{N1,T}, ip, I::NTuple{N2,Int64}
) where {N1,N2,T}
    quote
        Base.@_inline_meta
        Base.Cartesian.@nexprs $N1 i -> @cell p[i][ip, I...] = NaN
    end
end

@inline function fill_particle!(
    p::NTuple{N,T1}, field::NTuple{N,T2}, ip, I::Int64
) where {N,T1,T2}
    return fill_particle!(p, field, ip, (I,))
end

@generated function fill_particle!(
    p::NTuple{N1,T1}, field::NTuple{N1,T2}, ip, I::NTuple{N2,Int64}
) where {N1,N2,T1,T2}
    quote
        Base.Cartesian.@nexprs $N1 i -> begin
            Base.@_inline_meta
            tmp = p[i]
            @cell tmp[ip, I...] = field[i]
        end
        return nothing
    end
end

function clean_particles!(particles::Particles, grid, args)
    (; coords, index) = particles
    dxi = compute_dx(grid)
    ni = size(index)
    @parallel (@idx ni) _clean!(coords, grid, dxi, index, args)
    return nothing
end

@parallel_indices (i, j) function _clean!(
    particle_coords::NTuple{2,Any}, grid::NTuple{2,Any}, dxi::NTuple{2,Any}, index, args
)
    clean_kernel!(particle_coords, grid, dxi, index, args, i, j)
    return nothing
end

@parallel_indices (i, j, k) function _clean!(
    particle_coords::NTuple{3,Any}, grid::NTuple{3,Any}, dxi::NTuple{3,Any}, index, args
)
    clean_kernel!(particle_coords, grid, dxi, index, args, i, j, k)
    return nothing
end

function clean_kernel!(
    particle_coords, grid, dxi, index, args, cell_indices::Vararg{Int,N}
) where {N}
    corner_xi = corner_coordinate(grid, cell_indices...)
    # iterate over particles in child cell 
    for ip in cellaxes(index)
        pᵢ = cache_particle(particle_coords, ip, cell_indices)

        if @cell index[ip, cell_indices...] # true if memory allocation is filled with a particle
            if !(isincell(pᵢ, corner_xi, dxi))
                # remove particle from child cell
                @cell index[ip, cell_indices...] = false
                empty_particle!(particle_coords, ip, cell_indices)
                empty_particle!(args, ip, cell_indices)
            end
        end
    end
    return nothing
end

function global_domain_limits(origin::NTuple{N,Any}, dxi::NTuple{N,Any}) where {N}
    fn = nx_g, ny_g, nz_g

    lims = ntuple(Val(N)) do i
        Base.@_inline_meta
        origin[i], (fn[i]() - 1) * dxi[i]
    end

    return lims
end


# The following kernels are used in the `move_particles!` function
# to remove a random particle from the memory location so that the
# cell capacity is alwasy below 80% of its maximum.
@parallel_indices (I...) function empty_particles!(coords, index, cell_length, args)
    empty_kernel!(coords, index, cell_length, args, I)
    return nothing
end

function empty_kernel!(
    coords,
    index,
    cell_length,
    args::NTuple{N2},
    I::NTuple{N1,Int64},
) where {N1, N2}

    # count number of active particles inside I-th cell
    number_of_particles = count_particles(index, I...)
    # if the number of particles is less than 80% 
    # of the cell length then we do nothing
    max_particles_allowed = cell_length * 0.75
    number_of_particles < max_particles_allowed && return nothing

    # else we randomly remove particles until we are below 80% capacity
    number_of_particles_to_remove = number_of_particles - round(Int, max_particles_allowed, RoundDown)
    counter = 0
    while counter < number_of_particles_to_remove
        # randomly select a particle to remove
        index_to_remove = rand(1:number_of_particles)
        # check if a particle is actually in that memory location
        doskip(index, index_to_remove, I...) && continue
        # great, lets get rid of it
        @cell index[index_to_remove, I...] = false
        empty_particle!(coords, index_to_remove, I)
        empty_particle!(args, index_to_remove, I)
        counter += 1
    end
    return nothing
end

function count_particles(index, I::Vararg{Int,N}) where {N}
    count = 0
    for i in cellaxes(index)
        @inbounds count += @cell index[i, I...]
    end
    return count
end

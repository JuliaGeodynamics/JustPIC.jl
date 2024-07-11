using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

function empty_particles!(particles::AbstractParticles,  args)
    # implementation goes here
    (; coords, index, max_xcell) = particles
    nxi = size(index)

    @parallel (JustPIC._2D.@idx nxi) empty_particles!(coords, index, max_xcell, args)
    return nothing
end

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
    max_particles_allowed = cell_length * 0.8
    number_of_particles < max_particles_allowed && return nothing

    # else we randomly remove particles until we are below 80% capacity
    number_of_particles_to_remove = number_of_particles - round(Int, max_particles_allowed, RoundDown)
    counter = 0
    while counter < number_of_particles_to_remove
        # randomly select a particle to remove
        index_to_remove = rand(1:number_of_particles)
        # check if a particle is actually in that memory location
        doskip(index, index_to_remove, idx...) && continue
        # great, lets get rid of it
        @cell index[index_to_remove, I...] = false
        JustPIC._2D.empty_particle!(coords, index_to_remove, I)
        JustPIC._2D.empty_particle!(args, index_to_remove, I)

        counter += 14
    end
end

function count_particles(index, I::Vararg{Int,N}) where {N}
    count = 0
    for i in cellaxes(index)
        @inbounds count += @cell index[i, I...]
    end
    return count
end

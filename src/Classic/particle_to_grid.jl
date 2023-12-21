## LAUNCHERS

function particle2grid_naive!(F, Fp, buffer, xi::NTuple, particles::ClassicParticles)
    (; coords, parent_cell) = particles
    np = nparticles(particles)
    ni = size(F)

    @parallel (@idx ni) reset_arrays!(F, buffer)
    # accumulate weights on F and buffer arrays
    @parallel (@idx np) particle2grid_naive!(F, Fp, buffer, xi, coords, parent_cell)
    # finish interpolation process
    @parallel (@idx ni) resolve_particle2grid!(F, buffer)

    return nothing
end

@parallel_indices (ipart) function particle2grid_naive!(
    F, Fp, buffer, xi, coords, parent_cell
)
    _particle2grid_naive!(F, Fp, buffer, ipart, xi, coords, parent_cell[ipart]...)
    return nothing
end

## INTERPOLATION KERNEL 2D

@inbounds function _particle2grid_naive!(
    F, Fp, buffer, ipart, xi::NTuple{2,T}, coords, inode, jnode
) where {T}
    p_i = coords[ipart][1], coords[ipart][2] # particle coordinates
    # iterate over cells around i-th node
    for joffset in 0:1
        jvertex = joffset + jnode
        # make sure we stay within the grid
        for ioffset in 0:1
            ivertex = ioffset + inode
            xvertex = xi[1][ivertex], xi[2][jvertex] # cell lower-left coordinates
            # F acting as buffer here
            @myatomic F[ivertex, jvertex] += ω_i = distance_weight(xvertex, p_i; order=4)
            @myatomic buffer[ivertex, jvertex] += Fp[ipart] * ω_i
        end
    end

    return nothing
end

@parallel_indices (I...) function resolve_particle2grid!(F, buffer)
    @inbounds F[I...] = buffer[I...] * inv(F[I...])
    return nothing
end

@parallel_indices (I...) function reset_arrays!(F, buffer)
    @inbounds F[I...] = 0.0
    @inbounds buffer[I...] = 0.0
    return nothing
end

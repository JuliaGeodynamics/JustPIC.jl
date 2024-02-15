## LAUNCHERS

function particle2grid!(F, Fp, buffer, xi, particles::PassiveMarkers)
    (; coords, np) = particles
    ni = size(F)
    dxi = grid_size(xi)

    @parallel (@idx ni) reset_arrays!(F, buffer)
    # accumulate weights on F and buffer arrays
    @parallel (@idx np) passivemarker2grid!(F, Fp, buffer, xi, coords, dxi)
    # finish interpolation process
    @parallel (@idx ni) resolve_particle2grid!(F, buffer)

    return nothing
end

@parallel_indices (ipart) function passivemarker2grid!(
    F, Fp, buffer, xi, coords, dxi
)
    _passivemarker2grid!(F, Fp, buffer, ipart, xi, coords, dxi)
    return nothing
end

## INTERPOLATION KERNEL 2D

@inbounds function _passivemarker2grid!(
    F, Fp, buffer, ipart, xi::NTuple{2,T}, coords, dxi
) where {T}
    
    pᵢ = get_particle_coords(coords, ipart)
    
    inode, jnode = ntuple(Val(2)) do i
        Base.@_inline_meta
        cell_index(pᵢ[i], xi[i], dxi[i])
    end
    # iterate over cells around i-th node
    Fp_ipart = Fp[ipart]

    for joffset in 0:1
        jvertex = joffset + jnode
        # make sure we stay within the grid
        for ioffset in 0:1
            ivertex = ioffset + inode
            xvertex = xi[1][ivertex], xi[2][jvertex] # cell lower-left coordinates
            # F acting as buffer here
            @myatomic F[ivertex, jvertex] += ω_i = distance_weight(xvertex, pᵢ; order=4)
            @myatomic buffer[ivertex, jvertex] += Fp_ipart * ω_i
        end
    end

    return nothing
end

@parallel_indices (I...) function resolve_particle2grid!(F, buffer)
    @inbounds F[I...] = buffer[I...] * inv(F[I...])
    return nothing
end

@parallel_indices (I...) function reset_arrays!(A, B)
    @inbounds A[I...] = B[I...] = 0.0
    return nothing
end

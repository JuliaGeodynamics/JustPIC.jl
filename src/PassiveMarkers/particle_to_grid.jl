## LAUNCHERS

function particle2grid!(F, Fp, buffer, xi, particles::PassiveMarkers)
    (; coords, np) = particles
    ni = size(F)
    dxi = grid_size(xi)

    launch!(ka_backend(F), reset_arrays!, ni, F, buffer)
    # accumulate weights on F and buffer arrays
    launch!(ka_backend(F), passivemarker2grid!, np, F, Fp, buffer, xi, coords, dxi)
    # finish interpolation process
    launch!(ka_backend(F), resolve_particle2grid!, ni, F, buffer)

    return nothing
end

@kernel function passivemarker2grid!(F, Fp, buffer, xi, coords, dxi)
    ipart = @index(Global)
    _passivemarker2grid!(F, Fp, buffer, ipart, xi, coords, dxi)
end

## INTERPOLATION KERNEL 2D

@inbounds function _passivemarker2grid!(
        F, Fp, buffer, ipart, xi::NTuple{2}, coords, dxi
    )
    pᵢ = get_particle_coords(coords, ipart)
    # pᵢ = coords[ipart].data

    inode, jnode = ntuple(Val(2)) do i
        Base.@_inline_meta
        @inbounds cell_index(pᵢ[i], xi[i], dxi[i])
    end
    # iterate over cells around i-th node
    Fp_ipart = Fp[ipart]

    xv, yv = xi
    for joffset in 0:1
        jvertex = joffset + jnode
        # make sure we stay within the grid
        @inbounds for ioffset in 0:1
            ivertex = ioffset + inode
            xvertex = xv[ivertex], yv[jvertex] # cell lower-left coordinates
            # F acting as buffer here
            # ω_i = distance_weight(xvertex, pᵢ; order=4)
            ω_i = bilinear_weight(xvertex, pᵢ, dxi)
            KernelAbstractions.@atomic F[ivertex, jvertex] += ω_i
            KernelAbstractions.@atomic buffer[ivertex, jvertex] += Fp_ipart * ω_i
        end
    end

    return nothing
end

@kernel function resolve_particle2grid!(F, buffer)
    I = @index(Global, NTuple)
    @inbounds F[I...] = buffer[I...] * inv(F[I...])
end

@kernel function reset_arrays!(A, B)
    I = @index(Global, NTuple)
    @inbounds A[I...] = B[I...] = 0.0
end

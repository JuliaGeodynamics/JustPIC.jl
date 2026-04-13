## LAUNCHERS

"""
    particle2grid!(F, Fp, particles)

Interpolate particle-centered values `Fp` onto the grid nodes `F`.

The operation is performed in place and supports both scalar fields and tuples
of component arrays.

# Arguments
- `F`: destination nodal array, or tuple of nodal arrays.
- `Fp`: particle field stored with the same cell layout as `particles`.
- `particles`: the `Particles` container supplying particle coordinates and
  active-slot information. Its stored `xvi` coordinates define the target
  vertex grid.

# Notes
- This routine mutates `F` in place.
- The interpolation loops over nodes and nearby cells, which avoids atomics on
  the particles themselves.
"""

function particle2grid!(F, Fp, particles; ghost_1 = true, ghost_2 = true, ghost_3 = true)
    (; coords, index, xvi) = particles

    # mask shift in case `F` has ghost nodes only in some dimensions, or non at all
    mask = inner_mask(particles, ghost_1, ghost_2, ghost_3)

    @parallel (@idx inner_size(F)) particle2grid!(F, Fp, xvi, coords, index, mask)
    return nothing
end

@parallel_indices (I...) function particle2grid!(F, Fp, xi, particle_coords, index, mask)
    I_inner = I .+ 1
    _particle2grid!(F, Fp, I_inner..., xi, particle_coords, index, mask)
    return nothing
end

## INTERPOLATION KERNEL 2D

function _particle2grid!(F, Fp, inode, jnode, xi::NTuple{2, T}, p, index, mask) where {T}
    px, py = p # particle coordinates
    xvertex = xi[1][inode], xi[2][jnode] # cell lower-left coordinates
    ω, ωxF = 0.0, 0.0 # init weights

    # iterate over cells around i-th node
     for joffset in -1:0
        jvertex = joffset + jnode
        # !(1 ≤ jvertex < size(F, 2)) && continue # not needed because of ghost nodes

        # make sure we stay within the grid
        for ioffset in -1:0
            ivertex = ioffset + inode
            # !(1 ≤ ivertex < size(F, 1)) && continue # not needed because of ghost nodes

            # make sure we stay within the grid
            # iterate over cell
            for ip in cellaxes(px)
                # early exit if particle is not in the cell
                doskip(index, ip, ivertex, jvertex) && continue

                p_i = @index(px[ip, ivertex, jvertex]), @index(py[ip, ivertex, jvertex])
                ω_i = distance_weight(xvertex, p_i; order = 2)
                # @show ω_i
                # error()

                # ω_i = bilinear_weight(xvertex, p_i, di)
                ω += ω_i
                ωxF = fma(ω_i, @index(Fp[ip, ivertex, jvertex]), ωxF)
            end
        end
    end
     F[(inode, jnode).+mask...] = ωxF / ω
    return nothing
end

 function _particle2grid!(
        F::NTuple{N, T1}, Fp::NTuple{N, T2}, inode, jnode, xi::NTuple{2, T3}, p, index, mask
    ) where {N, T1, T2, T3}
    px, py = p # particle coordinates
    nx, ny = size(F[1])
    xvertex = xi[1][inode], xi[2][jnode] # cell lower-left coordinates
    ω, ωxF = 0.0, 0.0 # init weights

    # iterate over cells around i-th node
    for joffset in -1:0
        jvertex = joffset + jnode
        # make sure we stay within the grid
        for ioffset in -1:0
            ivertex = ioffset + inode
            # make sure we stay within the grid
            # if (1 ≤ ivertex < nx) && (1 ≤ jvertex < ny) # not needed because of ghost nodes
                # iterate over cell
                for i in cellaxes(px)
                    # ignore lines below for unused allocations
                    doskip(index, i, ivertex, jvertex) && continue

                    p_i = @index(px[i, ivertex, jvertex]), @index(py[i, ivertex, jvertex])
                    ω_i = distance_weight(xvertex, p_i; order = 2)
                    # ω_i = bilinear_weight(xvertex, p_i, di)
                    ω += ω_i
                    ωxF = ntuple(Val(N)) do j
                        Base.@_inline_meta
                        muladd(ω_i, @index(Fp[j][i, ivertex, jvertex]), ωxF[j])
                    end
                end
            # end
        end
    end

    _ω = inv(ω)
    return ntuple(Val(N)) do i
        Base.@_inline_meta
        F[i][(inode, jnode).+mask...] = ωxF[i] * _ω
    end
end

## INTERPOLATION KERNEL 3D

 function _particle2grid!(
        F, Fp, inode, jnode, knode, xi::NTuple{3, T}, p, index, mask
    ) where {T}
    px, py, pz = p # particle coordinates
    nx, ny, nz = size(F)
    xvertex = xi[1][inode], xi[2][jnode], xi[3][knode] # cell lower-left coordinates
    ω, ωF = 0.0, 0.0 # init weights

    # iterate over cells around i-th node
    for koffset in -1:0
        kvertex = koffset + knode
        for joffset in -1:0
            jvertex = joffset + jnode
            for ioffset in -1:0
                ivertex = ioffset + inode
                # make sure we operate within the grid
                # if (1 ≤ ivertex < nx) && (1 ≤ jvertex < ny) && (1 ≤ kvertex < nz)
                    # iterate over cell
                     for ip in cellaxes(px)
                        # ignore lines below for unused allocations
                        doskip(index, ip, ivertex, jvertex, kvertex) && continue

                        p_i = (
                            @index(px[ip, ivertex, jvertex, kvertex]),
                            @index(py[ip, ivertex, jvertex, kvertex]),
                            @index(pz[ip, ivertex, jvertex, kvertex]),
                        )
                        ω_i = distance_weight(xvertex, p_i; order = 2)
                        # ω_i = bilinear_weight(xvertex, p_i, di)
                        ω += ω_i
                        ωF = muladd(ω_i, @index(Fp[ip, ivertex, jvertex, kvertex]), ωF)
                    end
                # end
            end
        end
    end

    return F[(inode, jnode, knode).+mask...] = ωF * inv(ω)
end

 function _particle2grid!(
        F::NTuple{N, T1}, Fp::NTuple{N, T2}, inode, jnode, knode, xi::NTuple{3, T3}, p, index, mask
    ) where {N, T1, T2, T3}
    px, py, pz = p # particle coordinates
    nx, ny, nz = size(F[1])
    xvertex = xi[1][inode], xi[2][jnode], xi[3][knode] # cell lower-left coordinates
    ω = 0.0 # init weights
    ωxF = ntuple(i -> 0.0, Val(N)) # init weights

    # iterate over cells around i-th node
    for koffset in -1:0
        kvertex = koffset + knode
        for joffset in -1:0
            jvertex = joffset + jnode
            for ioffset in -1:0
                ivertex = ioffset + inode
                # make sure we operate within the grid
                # if (1 ≤ ivertex < nx) && (1 ≤ jvertex < ny) && (1 ≤ kvertex < nz)
                    # iterate over cell
                     for ip in cellaxes(px)
                        # ignore lines below for unused allocations
                        doskip(index, ip, ivertex, jvertex, kvertex) && continue

                        p_i = (
                            @index(px[ip, ivertex, jvertex, kvertex]),
                            @index(py[ip, ivertex, jvertex, kvertex]),
                            @index(pz[ip, ivertex, jvertex, kvertex]),
                        )
                        ω_i = distance_weight(xvertex, p_i; order = 2)
                        # ω_i = bilinear_weight(xvertex, p_i, di)
                        ω += ω_i
                        ωxF = ntuple(Val(N)) do j
                            Base.@_inline_meta
                            muladd(ω_i, @index(Fp[j][i, ivertex, jvertex, kvertex]), ωxF[j])
                        end
                    end
                # end
            end
        end
    end

    _ω = inv(ω)
    return ntuple(Val(N)) do i
        Base.@_inline_meta
         F[i][(inode, jnode, knode).+mask...] = ωxF[i] * _ω
    end
end

## OTHERS

@inline function distance_weight(a, b; order::Int64 = 1)
    return inv(distance(a, b)^order)
end

@inline function distance_weight(x, y, b; order::Int64 = 1)
    return inv(distance((x, y), b)^order)
end

@generated function bilinear_weight(
        a::Union{NTuple{N, T}, SVector{N, T}},
        b::Union{NTuple{N, T}, SVector{N, T}},
        di::Union{NTuple{N, T}, SVector{N, T}},
    ) where {N, T}
    return quote
        Base.@_inline_meta
        one_T = val = one(T)
        Base.Cartesian.@nexprs $N i ->
         val *= muladd(-abs(a[i] - b[i]), inv(di[i]), one_T)
        return val
    end
end

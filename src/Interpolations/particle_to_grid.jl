## LAUNCHERS

function particle2grid!(F, Fp, xi, particles)
    (; coords, index) = particles
    @parallel (@idx size(F)) particle2grid!(F, Fp, xi, coords, index)
    return nothing
end

@parallel_indices (I...) function particle2grid!(F, Fp, xi, particle_coords, index)
    _particle2grid!(F, Fp, I..., xi, particle_coords, index)
    return nothing
end

## INTERPOLATION KERNEL 2D

function _particle2grid!(F, Fp, inode, jnode, xi::NTuple{2, T}, p, index) where {T}
    px, py = p # particle coordinates
    xvertex = xi[1][inode], xi[2][jnode] # cell lower-left coordinates
    ω, ωxF = 0.0, 0.0 # init weights

    # iterate over cells around i-th node
    @inbounds for joffset in -1:0
        jvertex = joffset + jnode
        !(1 ≤ jvertex < size(F, 2)) && continue

        # make sure we stay within the grid
        for ioffset in -1:0
            ivertex = ioffset + inode
            !(1 ≤ ivertex < size(F, 1)) && continue

            # make sure we stay within the grid
            # iterate over cell
            for ip in cellaxes(px)
                # early exit if particle is not in the cell
                doskip(index, ip, ivertex, jvertex) && continue

                p_i = @index(px[ip, ivertex, jvertex]), @index(py[ip, ivertex, jvertex])
                ω_i = distance_weight(xvertex, p_i; order = 2)
                # ω_i = bilinear_weight(xvertex, p_i, di)
                ω += ω_i
                ωxF = fma(ω_i, @index(Fp[ip, ivertex, jvertex]), ωxF)
            end
        end
    end
    @inbounds F[inode, jnode] = ωxF / ω
    return nothing
end

@inbounds function _particle2grid!(
        F::NTuple{N, T1}, Fp::NTuple{N, T2}, inode, jnode, xi::NTuple{2, T3}, p, index
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
            if (1 ≤ ivertex < nx) && (1 ≤ jvertex < ny)
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
            end
        end
    end

    _ω = inv(ω)
    return ntuple(Val(N)) do i
        Base.@_inline_meta
        F[i][inode, jnode] = ωxF[i] * _ω
    end
end

## INTERPOLATION KERNEL 3D

@inbounds function _particle2grid!(
        F, Fp, inode, jnode, knode, xi::NTuple{3, T}, p, index
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
                if (1 ≤ ivertex < nx) && (1 ≤ jvertex < ny) && (1 ≤ kvertex < nz)
                    # iterate over cell
                    @inbounds for ip in cellaxes(px)
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
                end
            end
        end
    end

    return F[inode, jnode, knode] = ωF * inv(ω)
end

@inbounds function _particle2grid!(
        F::NTuple{N, T1}, Fp::NTuple{N, T2}, inode, jnode, knode, xi::NTuple{3, T3}, p, index
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
                if (1 ≤ ivertex < nx) && (1 ≤ jvertex < ny) && (1 ≤ kvertex < nz)
                    # iterate over cell
                    @inbounds for ip in cellaxes(px)
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
                end
            end
        end
    end

    _ω = inv(ω)
    return ntuple(Val(N)) do i
        Base.@_inline_meta
        @inbounds F[i][inode, jnode, knode] = ωxF[i] * _ω
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
        @inbounds val *= muladd(-abs(a[i] - b[i]), inv(di[i]), one_T)
        return val
    end
end

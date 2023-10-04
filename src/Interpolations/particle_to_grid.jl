## LAUNCHERS

function particle2grid!(F::AbstractArray, Fp::AbstractArray, xi::NTuple, particle_coords)
    dxi = grid_size(xi)
    particle2grid!(F, Fp, xi, particle_coords, dxi)
    return nothing
end

function particle2grid!(
    F::AbstractArray, Fp::AbstractArray, xi::NTuple, particle_coords, dxi
)
    @parallel (@idx size(F)) particle2grid!(F, Fp, xi, particle_coords, dxi)
    return nothing
end

@parallel_indices (I...) function particle2grid!(
    F, Fp, xi, particle_coords, di
) where {T}
    _particle2grid!(F, Fp, I, xi, particle_coords, di)
    return nothing
end

## INTERPOLATION KERNEL 2D

@inbounds function _particle2grid!(F, Fp, inode, jnode, xi::NTuple{2,T}, p, di) where {T}
    px, py = p # particle coordinates
    nx, ny = size(F)
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
                    p_i = @cell(px[i, ivertex, jvertex]), @cell(py[i, ivertex, jvertex])
                    # ignore lines below for unused allocations
                    any(isnan, p_i) && continue
                    ω_i = bilinear_weight(xvertex, p_i, di)
                    ω += ω_i
                    ωxF += ω_i * @cell(Fp[i, ivertex, jvertex])
                end
            end
        end
    end

    return F[inode, jnode] = ωxF / ω
end


@inbounds function _particle2grid!(F::NTuple{N,T1}, Fp::NTuple{N,T1}, inode, jnode, xi::NTuple{2,T2}, p, di) where {N,T1,T2}
    px, py = p # particle coordinates
    nx, ny = size(F)
    xvertex = xi[1][inode], xi[2][jnode] # cell lower-left coordinates
    ω = 0.0 # init weights
    ωxF = ntuple(i -> zero(T1), Val(N)) # init weights

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
                    p_i = @cell(px[i, ivertex, jvertex]), @cell(py[i, ivertex, jvertex])
                    # ignore lines below for unused allocations
                    any(isnan, p_i) && continue
                    ω_i = bilinear_weight(xvertex, p_i, di)
                    ω += ω_i
                    ωxF = ntuple(Val(N)) do j
                        Base.@_inline_meta
                        ωxF[j] + ω_i + @cell(Fp[j][i, ivertex, jvertex])
                    end
                end
            end
        end
    end

    _ω = inv(ω)
    return ntuple(Val(N)) do i
        Base.@_inline_meta
        F[i][inode, jnode] = ωxF * _ω
    end
end

## INTERPOLATION KERNEL 3D

@inbounds function _particle2grid!(
    F::NTuple{N,T1}, Fp::NTuple{N,T1}, inode, jnode, knode, xi::NTuple{3,T2}, p, di
) where {N,T1,T2}
    px, py, pz = p # particle coordinates
    nx, ny, nz = size(F)
    xvertex = xi[1][inode], xi[2][jnode], xi[3][knode] # cell lower-left coordinates
    ω = 0.0 # init weights
    ωxF = ntuple(i -> zero(T1), Val(N)) # init weights

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
                        p_i = (
                            @cell(px[ip, ivertex, jvertex, kvertex]),
                            @cell(py[ip, ivertex, jvertex, kvertex]),
                            @cell(pz[ip, ivertex, jvertex, kvertex]),
                        )
                        any(isnan, p_i) && continue  # ignore lines below for unused allocations
                        ω_i = bilinear_weight(xvertex, p_i, di)
                        ω += ω_i
                        ωxF = ntuple(Val(N)) do j
                            Base.@_inline_meta
                            ωxF[j] + ω_i + @cell(Fp[j][i, ivertex, jvertex, kvertex])
                        end
                    end
                end
            end
        end
    end

    _ω = inv(ω)
    return ntuple(Val(N)) do i
        Base.@_inline_meta
        F[i][inode, jnode, knode] = ωxF * _ω
    end
end

## OTHERS

@inline function distance_weight(a::NTuple{N,T}, b::NTuple{N,T}; order=2) where {N,T}
    return inv(distance(a, b)^order)
end

@generated function bilinear_weight(
    a::NTuple{N,T}, b::NTuple{N,T}, di::NTuple{N,T}
) where {N,T}
    quote
        Base.@_inline_meta
        val = one(T)
        Base.Cartesian.@nexprs $N i ->
            @inbounds val *= muladd(-abs(a[i] - b[i]), inv(di[i]), one(T))
        return val
    end
end

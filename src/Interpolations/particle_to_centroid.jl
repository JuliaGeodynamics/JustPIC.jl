## LAUNCHERS
"""
    particle2centroid!(F, Fp, xci::NTuple, particles::Particles)

Interpolate particle-centered values `Fp` to cell centers `F`.

`xci` contains the 1D coordinate arrays of the cell centers. This is the
cell-centered counterpart to `particle2grid!` and mutates `F` in place.
"""
particle2centroid!(F, Fp, particles::Particles) = particle2centroid!(F, Fp, particles.xci, particles, particles.di.vertex)

function particle2centroid!(F, Fp, xci::NTuple, particles::Particles, di)
    (; coords) = particles
    ndrange = length.(inner_range(coords[1]))
    launch!(ka_backend(particles), particle2centroid_kernel!, ndrange, F, Fp, xci, coords, di)
    return nothing
end

inner_range(A::AbstractArray{T, N}) where {T, N} = ntuple(i -> 2:(size(A, i) - 1), Val(N))

@kernel function particle2centroid_kernel!(F, Fp, xci, coords, di)
    I = @index(Global, NTuple)
    I_inner = I .+ 1
    _particle2centroid!(F, Fp, I_inner..., xci, coords, @dxi(di, I_inner...))
end

## INTERPOLATION KERNEL 2D

@inbounds function _particle2centroid!(
        F, Fp, inode, jnode, xci::NTuple{2, T}, p, di
    ) where {T}
    px, py = p # particle coordinates
    xcenter = xci[1][inode], xci[2][jnode] # centroid coordinates
    ω, ωxF = zero(eltype(F)), zero(eltype(F)) # init weights

    # iterate over cell
    for i in cellaxes(px)
        p_i = CAI.@index(px[i, inode, jnode]), CAI.@index(py[i, inode, jnode])
        # ignore lines below for unused allocations
        any(isnan, p_i) && continue
        ω_i = bilinear_weight(xcenter, p_i, di)
        # ω_i = distance_weight(xcenter, p_i; order=4)
        ω += ω_i
        # ωxF += ω_i * CAI.@index(Fp[i, inode, jnode])
        ωxF = muladd(ω_i, CAI.@index(Fp[i, inode, jnode]), ωxF)
    end

    return F[inode, jnode] = ωxF / ω
end

@inbounds function _particle2centroid!(
        F::NTuple{N, T1}, Fp::NTuple{N, T2}, inode, jnode, xci::NTuple{2, T3}, p, di
    ) where {N, T1, T2, T3}
    px, py = p # particle coordinates
    xcenter = xci[1][inode], xci[2][jnode] # centroid coordinates
    ω, ωxF = zero(eltype(F[1])), zero(eltype(F[1])) # init weights

    # iterate over cell
    for i in cellaxes(px)
        p_i = CAI.@index(px[i, inode, jnode]), CAI.@index(py[i, inode, jnode])
        # ignore lines below for unused allocations
        any(isnan, p_i) && continue
        # ω_i = bilinear_weight(xcenter, p_i, di)
        ω_i = distance_weight(xcenter, p_i; order = 2)

        ω += ω_i
        ωxF = ntuple(Val(N)) do j
            Base.@_inline_meta
            muladd(ω_i, CAI.@index(Fp[j][i, inode, jnode]), ωxF[j])
        end
    end

    _ω = inv(ω)
    return ntuple(Val(N)) do i
        Base.@_inline_meta
        F[i][inode, jnode] = ωxF[i] * _ω
    end
end

## INTERPOLATION KERNEL 3D

@inbounds function _particle2centroid!(
        F, Fp, inode, jnode, knode, xci::NTuple{3, T}, p, di
    ) where {T}
    px, py, pz = p # particle coordinates
    xcenter = xci[1][inode], xci[2][jnode], xci[3][knode] # centroid coordinates
    ω, ωF = zero(eltype(F)), zero(eltype(F)) # init weights

    # iterate over cell
    @inbounds for ip in cellaxes(px)
        p_i = (
            CAI.@index(px[ip, inode, jnode, knode]),
            CAI.@index(py[ip, inode, jnode, knode]),
            CAI.@index(pz[ip, inode, jnode, knode]),
        )
        isnan(p_i[1]) && continue  # ignore lines below for unused allocations
        ω_i = bilinear_weight(xcenter, p_i, di)
        ω += ω_i
        ωF = muladd(ω_i, CAI.@index(Fp[ip, inode, jnode, knode]), ωF)
    end

    return F[inode, jnode, knode] = ωF * inv(ω)
end

@inbounds function _particle2centroid!(
        F::NTuple{N, T1}, Fp::NTuple{N, T2}, inode, jnode, knode, xci::NTuple{3, T3}, p, di
    ) where {N, T1, T2, T3}
    px, py, pz = p # particle coordinates
    xcenter = xci[1][inode], xci[2][jnode], xci[3][knode] # centroid coordinates
    ω = zero(eltype(F[1])) # init weights
    ωxF = ntuple(i -> zero(eltype(F[1])), Val(N)) # init weights

    # iterate over cell
    @inbounds for ip in cellaxes(px)
        p_i = (
            CAI.@index(px[ip, inode, jnode, knode]),
            CAI.@index(py[ip, inode, jnode, knode]),
            CAI.@index(pz[ip, inode, jnode, knode]),
        )
        any(isnan, p_i) && continue  # ignore lines below for unused allocations
        ω_i = bilinear_weight(xcenter, p_i, di)
        ω += ω_i
        ωxF = ntuple(Val(N)) do j
            Base.@_inline_meta
            muladd(ω_i, CAI.@index(Fp[j][ip, inode, jnode, knode]), ωxF[j])
        end
    end

    _ω = inv(ω)
    return ntuple(Val(N)) do i
        Base.@_inline_meta
        F[i][inode, jnode, knode] = ωxF[i] * _ω
    end
end

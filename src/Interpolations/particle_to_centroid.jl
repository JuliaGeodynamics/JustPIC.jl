## LAUNCHERS
"""
    particle2centroid!(F, Fp, particles::Particles)
    particle2centroid!(F, Fp, xci::NTuple, particles::Particles, di)

Interpolate particle-centered values `Fp` to cell centers `F`.

`xci` contains the 1D coordinate arrays of the cell centers. This is the
cell-centered counterpart to `particle2grid!` and mutates `F` in place.

# Arguments
- `F`: destination centroid array, or tuple of centroid arrays.
- `Fp`: particle field stored with the same cell layout as `particles`.
- `particles`: the `Particles` container supplying particle coordinates. Its
  stored `xci` coordinates define the target centroid grid.
- `ghost_1`, `ghost_2`, `ghost_3`: whether `F` includes ghost nodes in each
  coordinate direction. Disable a keyword for a physical-only direction.
"""
particle2centroid!(F, Fp, particles::Particles; ghost_1 = true, ghost_2 = true, ghost_3 = true) =
    particle2centroid!(F, Fp, particles.xci, particles, particles.di.vertex; ghost_1 = ghost_1, ghost_2 = ghost_2, ghost_3 = ghost_3)

function particle2centroid!(F, Fp, xci::NTuple, particles::Particles, di; ghost_1 = true, ghost_2 = true, ghost_3 = true)
    (; coords) = particles
    backend = ka_backend(particles)
    Tc = eltype(eltype(coords[1]))
    xci = backend_grid(backend, xci, Tc)
    di = backend_grid(backend, di, Tc)

    # mask shift in case `F` has ghost nodes only in some dimensions, or non at all
    mask = inner_mask(particles, ghost_1, ghost_2, ghost_3)

    launch!(backend, particle2centroid_kernel!, inner_size(coords[1]), F, Fp, xci, coords, di, mask)
    return nothing
end

@kernel function particle2centroid_kernel!(F, Fp, xci, coords, di, mask)
    I = @index(Global, NTuple)
    I_inner = I .+ 1
    _particle2centroid!(F, Fp, I_inner, xci, coords, @dxi(di, I_inner...), mask)
end

## INTERPOLATION KERNEL 2D

@inbounds function _particle2centroid!(
        F, Fp, idx, xci::NTuple{2, T}, p, di, mask
    ) where {T}
    inode, jnode = idx
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

    return F[(inode, jnode) .+ mask...] = ωxF / ω
end

@inbounds function _particle2centroid!(
        F::NTuple{N, T1}, Fp::NTuple{N, T2}, idx, xci::NTuple{2, T3}, p, di, mask
    ) where {N, T1, T2, T3}
    inode, jnode = idx
    px, py = p # particle coordinates
    xcenter = xci[1][inode], xci[2][jnode] # centroid coordinates
    ω = zero(eltype(F[1])) # init weights
    ωxF = ntuple(i -> zero(eltype(F[1])), Val(N)) # init weights

    # iterate over cell
    for i in cellaxes(px)
        p_i = CAI.@index(px[i, inode, jnode]), CAI.@index(py[i, inode, jnode])
        # ignore lines below for unused allocations
        any(isnan, p_i) && continue
        # ω_i = bilinear_weight(xcenter, p_i, di)
        ω_i = distance_weight(xcenter, p_i; order = 2)

        ω += ω_i
        ωxF = let ωxF = ωxF, ω_i = ω_i
            ntuple(Val(N)) do j
                Base.@_inline_meta
                muladd(ω_i, CAI.@index(Fp[j][i, inode, jnode]), ωxF[j])
            end
        end
    end

    _ω = inv(ω)
    return ntuple(Val(N)) do i
        Base.@_inline_meta
        F[i][(inode, jnode) .+ mask...] = ωxF[i] * _ω
    end
end

## INTERPOLATION KERNEL 3D

@inbounds function _particle2centroid!(
        F, Fp, idx, xci::NTuple{3, T}, p, di, mask
    ) where {T}
    inode, jnode, knode = idx
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

    return F[(inode, jnode, knode) .+ mask...] = ωF * inv(ω)
end

@inbounds function _particle2centroid!(
        F::NTuple{N, T1}, Fp::NTuple{N, T2}, idx, xci::NTuple{3, T3}, p, di, mask
    ) where {N, T1, T2, T3}
    inode, jnode, knode = idx
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
        ωxF = let ωxF = ωxF, ω_i = ω_i
            ntuple(Val(N)) do j
                Base.@_inline_meta
                muladd(ω_i, CAI.@index(Fp[j][ip, inode, jnode, knode]), ωxF[j])
            end
        end
    end

    _ω = inv(ω)
    return ntuple(Val(N)) do i
        Base.@_inline_meta
        F[i][(inode, jnode, knode) .+ mask...] = ωxF[i] * _ω
    end
end

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
    check_transfer(Fp, F, ghosted_size(xci, (ghost_1, ghost_2, ghost_3)), particles)
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

    return F[(inode, jnode) .+ mask...] = ωxF * support_inverse(ω)
end

@inbounds function _particle2centroid!(
        F::NTuple{N, Any}, Fp::NTuple{N, Any}, idx, xci::NTuple{2, T3}, p, di, mask
    ) where {N, T3}
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

    _ω = support_inverse(ω)
    # `let` stops the closure from boxing the loop-reassigned `ωxF`; boxing allocates,
    # which GPU kernels cannot compile.
    return let ωxF = ωxF
        ntuple(Val(N)) do i
            Base.@_inline_meta
            F[i][(inode, jnode) .+ mask...] = ωxF[i] * _ω
        end
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

    return F[(inode, jnode, knode) .+ mask...] = ωF * support_inverse(ω)
end

@inbounds function _particle2centroid!(
        F::NTuple{N, Any}, Fp::NTuple{N, Any}, idx, xci::NTuple{3, T3}, p, di, mask
    ) where {N, T3}
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

    _ω = support_inverse(ω)
    # `let` stops the closure from boxing the loop-reassigned `ωxF`; boxing allocates,
    # which GPU kernels cannot compile.
    return let ωxF = ωxF
        ntuple(Val(N)) do i
            Base.@_inline_meta
            F[i][(inode, jnode, knode) .+ mask...] = ωxF[i] * _ω
        end
    end
end

## FLIP

"""
    particle2centroid_flip!(F, Fp, Fp0, particles; ghost_1 = true, ghost_2 = true, ghost_3 = true)

Add the interpolated particle increment `Fp - Fp0` to the cell centers `F`:
`F += Σ ω (Fp - Fp0) / Σ ω`, with the weights of [`particle2centroid!`](@ref).

`F`, `Fp` and `Fp0` may be single fields or tuples of fields. Cells with no
particles are left unchanged. See [`particle2grid_flip!`](@ref) for the
arguments.
"""
function particle2centroid_flip!(F, Fp, Fp0, particles; ghost_1 = true, ghost_2 = true, ghost_3 = true)
    (; coords, xci) = particles
    check_flip_transfer(Fp, Fp0, F, ghosted_size(xci, (ghost_1, ghost_2, ghost_3)), particles)
    backend = ka_backend(particles)
    Tc = eltype(eltype(coords[1]))
    xci = backend_grid(backend, xci, Tc)
    di = backend_grid(backend, particles.di.vertex, Tc)
    mask = inner_mask(particles, ghost_1, ghost_2, ghost_3)
    launch!(
        backend, particle2centroid_flip_kernel!, inner_size(coords[1]),
        as_tuple(F), as_tuple(Fp), as_tuple(Fp0), xci, coords, particles.index, di, mask
    )
    return nothing
end

@kernel function particle2centroid_flip_kernel!(F, Fp, Fp0, xci, coords, index, di, mask)
    I = @index(Global, NTuple)
    I_inner = I .+ 1
    _particle2centroid_flip!(F, Fp, Fp0, I_inner, xci, coords, index, @dxi(di, I_inner...), mask)
end

@inline function _particle2centroid_flip!(
        F::NTuple{NF}, Fp, Fp0, idx::NTuple{D}, xci, p, index, di, mask
    ) where {NF, D}
    xcenter = ntuple(d -> xci[d][idx[d]], Val(D))
    ω = zero(eltype(F[1]))
    acc = ntuple(_ -> zero(eltype(F[1])), Val(NF))

    for ip in cellaxes(p[1])
        doskip(index, ip, idx...) && continue
        p_i = get_particle_coords(p, ip, idx...)
        any(isnan, p_i) && continue
        ω_i = bilinear_weight(xcenter, p_i, di)
        ω += ω_i
        acc = _flip_accumulate(acc, ω_i, Fp, Fp0, ip, idx)
    end

    _flip_store!(F, acc, ω, idx .+ mask)
    return nothing
end

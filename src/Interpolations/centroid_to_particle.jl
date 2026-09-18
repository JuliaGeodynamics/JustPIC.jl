## CLASSIC PIC ------------------------------------------------------------------------------------------------

# LAUNCHERS
"""
    centroid2particle!(Fp, xci, F, particles)

Interpolate cell-centered field values `F` to particle values `Fp`.

`xci` contains the center coordinates of the grid carrying `F`. The destination
`Fp` is mutated in place and may be either a single particle field or a tuple of
particle fields.

Particles lying between a domain boundary and the first centroid are
interpolated from the ghost centroids, so `F` must always use the ghosted
`particles.xci` layout — unlike `grid2particle!`, there is no opt-out.
"""
function centroid2particle!(Fp, F, particles; ghosted = true)
    if ghosted
        centroid2particle_ghosted!(Fp, particles.xci, F, particles, particles.di.center)
    else
        xci = ntuple(i -> particles.xci[i][2:(end - 1)], Val(length(particles.xci)))
        di = ntuple(i -> diff(xci[i]), Val(length(xci)))
        centroid2particle_unghosted!(Fp, xci, F, particles, di)
    end
    return nothing
end

function centroid2particle!(Fp, xci, F, particles, di; ghosted = true)
    if ghosted
        centroid2particle_ghosted!(Fp, xci, F, particles, di)
    else
        centroid2particle_unghosted!(Fp, xci, F, particles, di)
    end
    return nothing
end

function centroid2particle_ghosted!(Fp, xci, F, particles, di)
    (; coords) = particles

    ni = inner_size(Fp)
    backend = ka_backend(particles)
    Tc = eltype(eltype(coords[1]))
    xci = backend_grid(backend, xci, Tc)
    di = backend_grid(backend, di, Tc)
    launch!(backend, centroid2particle_classic_ghosted!, ni, Fp, F, xci, di, coords)
    return nothing
end

function centroid2particle_unghosted!(Fp, xci, F, particles, di)
    (; coords) = particles

    backend = ka_backend(particles)
    Tc = eltype(eltype(coords[1]))
    xci = backend_grid(backend, xci, Tc)
    di = backend_grid(backend, di, Tc)
    launch!(backend, centroid2particle_classic_unghosted!, size(F), Fp, F, xci, di, coords)
    return nothing
end

@kernel function centroid2particle_classic_ghosted!(Fp, F, xci, di, coords)
    I = @index(Global, NTuple)
    _centroid2particle_classic_ghosted!(Fp, coords, xci, di, F, I .+ 1)
end

@kernel function centroid2particle_classic_unghosted!(Fp, F, xci, di, coords)
    I = @index(Global, NTuple)
    _centroid2particle_classic_unghosted!(Fp, coords, xci, di, F, I, I .+ 1)
end

# INNERMOST INTERPOLATION KERNEL

@inline function _centroid2particle_classic_ghosted!(Fp, p, xci, di::NTuple{N}, F, I) where {N}
    ni = size(F) .- 1
    # iterate over all the particles within the cells of index `idx`
    @inbounds for ip in cellaxes(Fp)
        # cache particle coordinates
        pᵢ = ntuple(i -> (CAI.@index p[i][ip, I...]), Val(N))
        # skip lines below if there is no particle in this piece of memory
        any(isnan, pᵢ) && continue
        # continue the kernel
        xc = ntuple(i -> xci[i][I[i]], Val(N))
        cell_index = shifted_index(pᵢ, xc, I)
        cell_index = clamp.(cell_index, 1, ni)
        # Interpolate field F onto particle
        # @show @dxi(di, cell_index...)
        CAI.@index Fp[ip, I...] = _grid2particle(pᵢ, xci, @dxi(di, cell_index...), F, cell_index)
    end
    return nothing
end

@inline function _centroid2particle_classic_unghosted!(Fp, p, xci, di::NTuple{N}, F, I_src, I_dst) where {N}
    ni = size(F) .- 1
    @inbounds for ip in cellaxes(Fp)
        pᵢ = ntuple(i -> (CAI.@index p[i][ip, I_dst...]), Val(N))
        any(isnan, pᵢ) && continue
        xc = ntuple(i -> xci[i][I_src[i]], Val(N))
        cell_index = shifted_index(pᵢ, xc, I_src)
        cell_index = clamp.(cell_index, 1, ni)
        CAI.@index Fp[ip, I_dst...] = _grid2particle(pᵢ, xci, @dxi(di, cell_index...), F, cell_index)
    end
    return nothing
end

@inline function _centroid2particle_classic_ghosted!(
        Fp::NTuple{NF}, p, xci, di::NTuple{N}, F::NTuple{NF}, I
    ) where {NF, N}
    ni = size(first(F)) .- 1
    # iterate over all the particles within the cells of index `idx`
    @inbounds for ip in cellaxes(Fp)
        # cache particle coordinates
        pᵢ = ntuple(i -> (CAI.@index p[i][ip, I...]), Val(N))
        # skip lines below if there is no particle in this piece of memory
        any(isnan, pᵢ) && continue
        # continue the kernel
        xc = ntuple(i -> xci[i][I[i]], Val(N))
        cell_index = shifted_index(pᵢ, xc, I)
        # cell_index = clamp.(cell_index, 1, ni) # no need with ghost nodes
        # Interpolate field F onto particle
        for n in 1:NF # should be unrolled
            CAI.@index Fp[n][ip, I...] = _grid2particle(pᵢ, xci, @dxi(di, cell_index...), F[n], cell_index)
        end
    end
    return nothing
end

@inline function _centroid2particle_classic_unghosted!(
        Fp::NTuple{NF}, p, xci, di::NTuple{N}, F::NTuple{NF}, I_src, I_dst
    ) where {NF, N}
    ni = size(first(F)) .- 1
    @inbounds for ip in cellaxes(Fp)
        pᵢ = ntuple(i -> (CAI.@index p[i][ip, I_dst...]), Val(N))
        any(isnan, pᵢ) && continue
        xc = ntuple(i -> xci[i][I_src[i]], Val(N))
        cell_index = shifted_index(pᵢ, xc, I_src)
        cell_index = clamp.(cell_index, 1, ni)
        for n in 1:NF
            CAI.@index Fp[n][ip, I_dst...] = _grid2particle(pᵢ, xci, @dxi(di, cell_index...), F[n], cell_index)
        end
    end
    return nothing
end

## UTILS ------------------------------------------------------------------------------------------------------

# shifts the index of the cell bot-left to the left if it is located in the left cell
@inline shifted_index(pxi, xci, idx) = pxi < xci ? idx - 1 : idx
@inline shifted_index(
    pxi::NTuple{N, A}, xci::NTuple{N, B}, idx::NTuple{N, Integer}
) where {N, A, B} = ntuple(i -> shifted_index(pxi[i], xci[i], idx[i]), Val(N))

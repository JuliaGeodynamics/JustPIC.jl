## CLASSIC PIC ------------------------------------------------------------------------------------------------

# LAUNCHERS
"""
    centroid2particle!(Fp, F, particles; ghosted = true)
    centroid2particle!(Fp, xci, F, particles, di; ghosted = true)

Interpolate cell-centered field values `F` to particle values `Fp`.

The destination `Fp` is mutated in place and may be either a single particle
field or a tuple of particle fields.

# Arguments
- `xci`, `di`: optional explicit centroid coordinates and spacing of the grid
  carrying `F`, for lower-level/internal use.
- `ghosted`: whether `F` uses the ghosted `particles.xci` layout. Particles
  lying between a domain boundary and the first centroid are then interpolated
  from the ghost centroids.
- `ghost_1`, `ghost_2`, `ghost_3`: per-direction override of `ghosted`
  (`centroid2particle!(Fp, F, particles)` only). When a direction is not
  ghosted, particles outside the first/last centroid use the nearest physical
  cell.
"""
function centroid2particle!(
        Fp, F, particles; ghosted = true, ghost_1 = ghosted, ghost_2 = ghosted, ghost_3 = ghosted
    )
    xci, di, off = centroid_layout(particles, (ghost_1, ghost_2, ghost_3))
    check_transfer(Fp, F, map(length, xci), particles)
    _centroid2particle!(Fp, xci, F, nothing, particles, di, off, 1)
    return nothing
end

function centroid2particle!(Fp, xci, F, particles, di; ghosted = true)
    check_transfer(Fp, F, map(length, xci), particles)
    _centroid2particle!(Fp, xci, F, nothing, particles, di, ntuple(_ -> Int(ghosted), Val(length(xci))), 1)
    return nothing
end

"""
    centroid2particle_flip!(Fp, F, F0, particles; α = 0.0, ghosted = true)

Update particle values from cell-centered fields with a PIC/FLIP blend.

`α = 1` gives pure PIC, `α = 0` gives pure FLIP: `Fp += F - F0`, with `F` and
`F0` interpolated to the particle. `Fp`, `F` and `F0` may be single fields or
tuples of fields. See [`centroid2particle!`](@ref) for the `ghosted`,
`ghost_1`, `ghost_2` and `ghost_3` keywords.
"""
function centroid2particle_flip!(
        Fp, F, F0, particles; α = 0.0, ghosted = true, ghost_1 = ghosted, ghost_2 = ghosted, ghost_3 = ghosted
    )
    0 ≤ α ≤ 1 || throw(ArgumentError("the PIC fraction `α` must lie in [0, 1], got $α"))
    xci, di, off = centroid_layout(particles, (ghost_1, ghost_2, ghost_3))
    dims = map(length, xci)
    check_transfer(Fp, F, dims, particles)
    check_field_pairing("F", F, "F0", F0)
    check_grid_field("F0", F0, dims, particles)
    check_distinct("Fp" => Fp, "F0" => F0)
    _centroid2particle!(Fp, xci, F, F0, particles, di, off, α)
    return nothing
end

# Centroid grid, spacing and source-index offset of `F` for the given ghost layout.
function centroid_layout(particles, ghosts)
    N = length(particles.xci)
    g = ntuple(i -> ghosts[i], Val(N))
    all(g) && return particles.xci, particles.di.center, ntuple(_ -> 1, Val(N))
    xci = ntuple(i -> g[i] ? particles.xci[i] : particles.xci[i][2:(end - 1)], Val(N))
    return xci, ntuple(i -> diff(xci[i]), Val(N)), map(Int, g)
end

# `F0 === nothing` gives plain interpolation, otherwise the PIC/FLIP blend with PIC fraction `α`.
function _centroid2particle!(Fp, xci, F, F0, particles, di, off, α)
    backend = ka_backend(particles)
    Tc = eltype(eltype(particles.coords[1]))
    xci = backend_grid(backend, xci, Tc)
    di = backend_grid(backend, di, Tc)
    F0 = isnothing(F0) ? F0 : as_tuple(F0)
    launch!(
        backend, centroid2particle_kernel!, inner_size(Fp), as_tuple(Fp), as_tuple(F), F0,
        xci, di, particles.coords, convert(Tc, α), off
    )
    return nothing
end

@kernel function centroid2particle_kernel!(Fp, F, F0, xci, di, coords, α, off)
    I = @index(Global, NTuple)
    _centroid2particle!(Fp, coords, xci, di, F, F0, α, I .+ off, I .+ 1)
end

# INNERMOST INTERPOLATION KERNEL
# `I_src` indexes the centroid grid of `F`, `I_dst` the particle cells.

@generated function _centroid2particle!(
        Fp::NTuple{NF}, p, xci, di::NTuple{N}, F::NTuple{NF}, F0, α, I_src, I_dst
    ) where {NF, N}
    value = if F0 <: Nothing
        :(lerp(field_corners(F[f], cell_index), ti))
    else
        :(
            _flip_blend(
                CAI.@index(Fp[f][ip, I_dst...]),
                lerp(field_corners(F[f], cell_index), ti),
                lerp(field_corners(F0[f], cell_index), ti),
                α,
            )
        )
    end
    return quote
        Base.@_inline_meta
        ni = size(F[1]) .- 1
        xc = Base.@ntuple $N i -> xci[i][I_src[i]]
        @inbounds for ip in cellaxes(Fp)
            pᵢ = get_particle_coords(p, ip, I_dst...)
            any(isnan, pᵢ) && continue
            cell_index = clamp.(shifted_index(pᵢ, xc, I_src), 1, ni)
            ti = normalize_coordinates(pᵢ, xci, @dxi(di, cell_index...), cell_index)
            Base.@nexprs $NF f -> CAI.@index Fp[f][ip, I_dst...] = $value
        end
        return nothing
    end
end

## UTILS ------------------------------------------------------------------------------------------------------

# shifts the index of the cell bot-left to the left if it is located in the left cell
@inline shifted_index(pxi, xci, idx) = pxi < xci ? idx - 1 : idx
@inline shifted_index(
    pxi::NTuple{N, Any}, xci::NTuple{N, Any}, idx::NTuple{N, Integer}
) where {N} = ntuple(i -> shifted_index(pxi[i], xci[i], idx[i]), Val(N))

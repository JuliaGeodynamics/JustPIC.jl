## CLASSIC PIC ------------------------------------------------------------------------------------------------

# LAUNCHERS

"""
    grid2particle!(Fp, F, particles)
    grid2particle!(Fp, xvi, F, particles, di)

Interpolate a nodal field `F` to particle values `Fp`.

`Fp` is updated in place for every active particle in `particles`.

# Arguments
- `Fp`: destination particle field, or tuple of particle fields.
- `F`: source nodal field.
- `particles`: particle container that provides positions and active slots.
- `xvi`: optional explicit vertex coordinates for lower-level/internal use.
- `di`: optional precomputed grid spacing for lower-level/internal use.
- `ghost_1`, `ghost_2`, `ghost_3`: whether `F` includes ghost nodes in each
  coordinate direction. Disable a keyword for a physical-only direction.

# Notes
- `Fp` must be allocated with the same cell layout as `particles.coords`.
- `grid2particle!(Fp, F, particles)` is the public convenience entry point and
  reads the vertex grid and spacing from `particles`.
"""

grid2particle!(Fp, F, particles; ghost_1 = true, ghost_2 = true, ghost_3 = true) = grid2particle!(Fp, particles.xvi, F, particles, particles.di.vertex; ghost_1 = ghost_1, ghost_2 = ghost_2, ghost_3 = ghost_3)

function grid2particle!(Fp, xvi, F, particles, di; ghost_1 = true, ghost_2 = true, ghost_3 = true)
    (; coords, index) = particles
    check_transfer(Fp, F, ghosted_size(xvi, (ghost_1, ghost_2, ghost_3)), particles)
    ni = inner_size(index)
    backend = ka_backend(particles)
    Tc = eltype(eltype(coords[1]))
    xvi = backend_grid(backend, xvi, Tc)
    di = backend_grid(backend, di, Tc)

    # mask shift in case `F` has ghost nodes only in some dimensions, or non at all
    mask = inner_mask(particles, ghost_1, ghost_2, ghost_3)

    launch!(backend, grid2particle_classic!, ni, Fp, F, xvi, index, di, coords, mask)

    return nothing
end


@kernel function grid2particle_classic!(
        Fp, F, xvi, index, di, particle_coords, mask
    )
    I = @index(Global, NTuple)
    I_inner = I .+ 1
    _grid2particle_classic!(
        Fp, particle_coords, xvi, @dxi(di, I_inner...), F, index, I_inner, Val(cellnum(index)), mask
    )
end

# INNERMOST INTERPOLATION KERNEL

@generated function _grid2particle_classic!(
        Fp, p, xvi, di, F, index, idx, ::Val{N}, mask
    ) where {N}
    return quote
        Base.@_inline_meta
        Fi = field_corners(F, idx .+ mask)
        xi_corner = corner_coordinate(xvi, idx)
        # iterate over all the particles within the cells of index `idx`
        Base.@nexprs $N ip -> begin
            # skip lines below if there is no particle in this piece of memory
            @inbounds if !doskip(index, ip, idx...)
                # cache particle coordinates
                pᵢ = get_particle_coords(p, ip, idx...)
                # Interpolate field F onto particle
                CAI.@index Fp[ip, idx...] = _grid2particle(pᵢ, xi_corner, di, Fi)
            end
        end
    end
end

@generated function _grid2particle_classic!(
        Fp::NTuple{NF}, p, xvi, di, F::NTuple{NF}, index, idx, ::Val{NP}, mask
    ) where {NF, NP}
    return quote
        Base.@_inline_meta
        Fi = Base.@ntuple $NF field -> field_corners(F[field], idx .+ mask)
        xi_corner = corner_coordinate(xvi, idx)
        Base.@nexprs $NP ip -> begin
            @inbounds if !doskip(index, ip, idx...)
                pᵢ = get_particle_coords(p, ip, idx...)
                Base.@nexprs $NF field -> begin
                    CAI.@index Fp[field][ip, idx...] = _grid2particle(
                        pᵢ, xi_corner, di, Fi[field]
                    )
                end
            end
        end
        return nothing
    end
end

## FULL PARTICLE PIC ------------------------------------------------------------------------------------------

# LAUNCHERS

"""
    grid2particle_flip!(Fp, F, F0, particles; α = 0.0)

Update particle values with a PIC/FLIP blend.

`α = 1` gives pure PIC, `α = 0` gives pure FLIP, and intermediate values blend
between the two updates.

# Arguments
- `Fp`: particle field (or tuple of particle fields) to update in place.
- `F`: current grid field (or tuple of fields).
- `F0`: previous grid field (or tuple of fields), same layout as `F`.
- `particles`: particle container; its vertex grid `particles.xvi` is used.
- `α`: PIC fraction in the PIC/FLIP blend, in `[0, 1]`.
- `ghost_1`, `ghost_2`, `ghost_3`: whether `F` and `F0` include ghost nodes in
  each coordinate direction. Disable a keyword for a physical-only direction.

!!! note
    The older method `grid2particle_flip!(Fp, xvi, F, F0, particles; ...)` that takes
    the vertex coordinates explicitly is deprecated; use the method above.
"""
grid2particle_flip!(Fp, F, F0, particles; α = 0.0, ghost_1 = true, ghost_2 = true, ghost_3 = true) =
    _grid2particle_flip!(Fp, particles.xvi, F, F0, particles; α = α, ghost_1 = ghost_1, ghost_2 = ghost_2, ghost_3 = ghost_3)

function grid2particle_flip!(Fp, xvi, F, F0, particles; kwargs...)
    Base.depwarn(
        "`grid2particle_flip!(Fp, xvi, F, F0, particles)` is deprecated; use `grid2particle_flip!(Fp, F, F0, particles)`",
        :grid2particle_flip!,
    )
    return _grid2particle_flip!(Fp, xvi, F, F0, particles; kwargs...)
end

function _grid2particle_flip!(Fp, xvi, F, F0, particles; α = 0.0, ghost_1 = true, ghost_2 = true, ghost_3 = true)
    (; coords, index) = particles
    0 ≤ α ≤ 1 || throw(ArgumentError("the PIC fraction `α` must lie in [0, 1], got $α"))
    dims = ghosted_size(xvi, (ghost_1, ghost_2, ghost_3))
    check_transfer(Fp, F, dims, particles)
    check_field_pairing("F", F, "F0", F0)
    check_grid_field("F0", F0, dims, particles)
    check_distinct("Fp" => Fp, "F0" => F0)
    # recast the grid to the particle precision so the ranges are GPU-safe on Float32
    # backends (they are indexed directly inside the kernel; see advection!)
    Tc = eltype(eltype(coords[1]))
    xvi = recast_grid(xvi, Tc)
    di = grid_size(xvi)
    di = backend_grid(ka_backend(particles), di, Tc)
    ni = inner_size(index)
    # blend factor must match the field precision (Float64 breaks Metal)
    αT = convert(Tc, α)

    # mask shift in case `F` has ghost nodes only in some dimensions, or non at all
    mask = inner_mask(particles, ghost_1, ghost_2, ghost_3)

    launch!(ka_backend(particles), grid2particle_full!, ni, Fp, F, F0, xvi, di, coords, index, αT, mask)

    return nothing
end

@kernel function grid2particle_full!(
        Fp, F, F0, xvi, di, particle_coords, index, α, mask
    )
    I = @index(Global, NTuple)
    I_inner = I .+ 1
    _grid2particle_full!(Fp, particle_coords, xvi, @dxi(di, I_inner...), F, F0, index, I_inner, α, mask)
end

# INNERMOST INTERPOLATION KERNEL

# PIC value `F_pic` blended with the FLIP update `Fp_old + (F_pic - F0_pic)`
@inline function _flip_blend(Fp_old, F_pic, F0_pic, α)
    return muladd(F_pic, α, (Fp_old + (F_pic - F0_pic)) * (one(α) - α))
end

@inline function _grid2particle_full!(
        Fp, p, xvi, di, F, F0, index, idx, α, mask
    )
    Fi = field_corners(F, idx .+ mask)
    F0i = field_corners(F0, idx .+ mask)

    # iterate over all the particles within the cells of index `idx`
    return @inbounds for ip in cellaxes(Fp)
        # skip lines below if there is no particle in this piece of memory
        doskip(index, ip, idx...) && continue

        pᵢ = get_particle_coords(p, ip, idx...)
        ti = normalize_coordinates(pᵢ, xvi, di, idx)
        Fᵢ = CAI.@index Fp[ip, idx...]
        CAI.@index Fp[ip, idx...] = _flip_blend(Fᵢ, lerp(Fi, ti), lerp(F0i, ti), α)
    end
end

@generated function _grid2particle_full!(
        Fp::NTuple{NF}, p, xvi, di, F::NTuple{NF}, F0::NTuple{NF}, index, idx, α, mask
    ) where {NF}
    return quote
        Base.@_inline_meta
        Fi = Base.@ntuple $NF f -> field_corners(F[f], idx .+ mask)
        F0i = Base.@ntuple $NF f -> field_corners(F0[f], idx .+ mask)
        @inbounds for ip in cellaxes(Fp)
            doskip(index, ip, idx...) && continue
            pᵢ = get_particle_coords(p, ip, idx...)
            ti = normalize_coordinates(pᵢ, xvi, di, idx)
            Base.@nexprs $NF f -> begin
                Fᵢ = CAI.@index Fp[f][ip, idx...]
                CAI.@index Fp[f][ip, idx...] = _flip_blend(
                    Fᵢ, lerp(Fi[f], ti), lerp(F0i[f], ti), α
                )
            end
        end
        return nothing
    end
end

#  Interpolation from grid corners to particle positions --------------------------------------------------------

@inline function _grid2particle(
        pᵢ::Union{SVector, NTuple}, xvi::NTuple, di::NTuple, F::AbstractArray, idx
    )
    # F at the cell corners
    Fi = field_corners(F, idx)
    # normalize particle coordinates
    ti = normalize_coordinates(pᵢ, xvi, di, idx)
    # Interpolate field F onto particle
    Fp = lerp(Fi, ti)

    return Fp
end

@inline _grid2particle(pᵢ::Union{SVector, NTuple}, xvi::NTuple, di::NTuple, ::Tuple{}, idx) = ()

@inline function _grid2particle(
        pᵢ::Union{SVector, NTuple}, xvi::NTuple, di::NTuple, Fi::NTuple{N, Number}, idx
    ) where {N}
    # normalize particle coordinates
    ti = normalize_coordinates(pᵢ, xvi, di, idx)
    # Interpolate field F onto particle
    Fp = lerp(Fi, ti)

    return Fp
end

@inline function _grid2particle(
        pᵢ::Union{SVector, NTuple}, xvi::NTuple, di::NTuple, F::NTuple{N, AbstractArray}, idx
    ) where {N}
    # normalize particle coordinates
    ti = normalize_coordinates(pᵢ, xvi, di, idx)
    Fp = ntuple(Val(N)) do i
        Base.@_inline_meta
        # F at the cell corners
        Fi = field_corners(F[i], idx)
        # Interpolate field F onto particle
        lerp(Fi, ti)
    end

    return Fp
end

@inline function _grid2particle(
        pᵢ::Union{SVector, NTuple}, xvi::NTuple, di::NTuple, F::NTuple{N1, Tuple{Vararg{Number}}}, idx
    ) where {N1}
    # normalize particle coordinates
    ti = normalize_coordinates(pᵢ, xvi, di, idx)
    Fp = ntuple(Val(N1)) do i
        Base.@_inline_meta
        # Interpolate field F onto particle
        lerp(F[i], ti)
    end

    return Fp
end

@inline function _grid2particle(
        pᵢ::Union{SVector, NTuple}, xvi::NTuple{N1, Any}, di::NTuple, Fi::NTuple{N2, Any}
    ) where {N1, N2}
    # normalize particle coordinates
    ti = normalize_coordinates(pᵢ, xvi, di)
    # Interpolate field F onto particle
    Fp = lerp(Fi, ti)
    return Fp
end

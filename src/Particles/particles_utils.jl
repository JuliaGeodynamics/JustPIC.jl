@inline function init_cell_arrays(particles::Particles, ::Val{N}) where {N}
    return ntuple(
        _ -> @fill(
            0.0, size(particles.coords[1])..., celldims = (cellsize(particles.index))
        ),
        Val(N),
    )
end

@inline function cell_array(
        x::T, ncells::NTuple{N1, Integer}, ni::NTuple{N2, Integer}
    ) where {T, N1, N2}
    return @fill(x, ni..., celldims = ncells, eltype = T)
end

## random particles initialization
"""
    init_particles( backend, nxcell, max_xcell, min_xcell, coords::NTuple{N,AbstractArray}, dxᵢ::NTuple{N,T}, nᵢ::NTuple{N,I})

Initialize the particles object.

# Arguments
- `backend`: Device backend
- `nxcell`: Initial number of particles per cell
- `max_xcell`: Maximum number of particles per cell
- `min_xcell`: Minimum number of particles per cell
- `xvi`: Grid cells vertices
"""
function init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi::Vararg{N, T}
    ) where {N, T}
    di = compute_dx(xvi)
    ni = @. length(xvi) - 1

    return _init_particles(backend, nxcell, max_xcell, min_xcell, xvi, di, ni)
end

# random distribution
function _init_particles(
        backend,
        nxcell::Number,
        max_xcell,
        min_xcell,
        coords::NTuple{N, AbstractArray},
        dxᵢ::NTuple{N},
        nᵢ::NTuple{N, I}
    ) where {N, I}

    # number of particles per quadrant
    NQ = N == 2 ? 4 : 8
    np_quadrant = ceil(Int, nxcell / NQ)
    nxcell = np_quadrant * NQ
    max_xcell = max(nxcell, max_xcell)
    np = max_xcell * prod(nᵢ)
    pxᵢ = ntuple(_ -> @fill(NaN, nᵢ..., celldims = (max_xcell,)), Val(N))
    index = @fill(false, nᵢ..., celldims = (max_xcell,), eltype = Bool)

    @parallel (@idx nᵢ) fill_coords_index(
        pxᵢ, index, coords, dxᵢ, np_quadrant
    )

    return Particles(backend, pxᵢ, index, nxcell, max_xcell, min_xcell, np)
end

@parallel_indices (I...) function fill_coords_index(
        pxᵢ::NTuple{N, T}, index, coords, di::NTuple{N}, np_quadrant
    ) where {N, T}
    # lower-left corner of the cell
    x0ᵢ = ntuple(Val(N)) do ndim
        @inline
        coords[ndim][I[ndim]]
    end
    dxᵢ = @dxi di I...
    masks = quadrant_masks(Val(N))
    # fill index array
    l = 0 # particle counter
    for iq in eachindex(masks)
        xcᵢ = x0ᵢ .+ dxᵢ .* 0.5 .* masks[iq] # quadrant lower-left coordinates
        for _ in 1:np_quadrant
            l += 1
            for ndim in 1:N
                @index pxᵢ[ndim][l, I...] = xcᵢ[ndim] + dxᵢ[ndim] / 2 * rand()
            end
            @index index[l, I...] = true
        end
    end
    return nothing
end

# regular distribution of markers
function _init_particles(
        backend,
        nxdim::NTuple{N, Integer},
        max_xcell,
        min_xcell,
        coords::NTuple{N, AbstractArray},
        dxᵢ::NTuple{N},
        nᵢ::NTuple{N, I}
    ) where {N, I}
    nxcell = prod(nxdim)
    ncells = prod(nᵢ)
    np = max_xcell * ncells
    pxᵢ = ntuple(_ -> @fill(NaN, nᵢ..., celldims = (max_xcell,)), Val(N))
    index = @fill(false, nᵢ..., celldims = (max_xcell,), eltype = Bool)

    di = compute_dx(coords)
    # offsets = ntuple(i -> LinRange(0, dxi[i], nxdim[i] + 2)[2:(end - 1)], Val(N))

    @parallel_indices (I...) function fill_coords_index(
            pxᵢ::NTuple{N, T}, index, coords, nxdim, di
        ) where {N, T}

        dxi = @dxi di I...

        # lower-left corner of the cell
        x0ᵢ = ntuple(Val(N)) do ndim
            coords[ndim][I[ndim]]
        end

        offsets = ntuple(i -> LinRange(0, dxi[i], nxdim[i] + 2)[2:(end - 1)], Val(N))
        
        # fill index array
        if N == 2

            for i in axes(offsets[1], 1), j in axes(offsets[2], 1)
                l = i + (j - 1) * nxdim[1]
                ndim = 1
                @index pxᵢ[ndim][l, I...] = x0ᵢ[ndim] + offsets[ndim][i]
                ndim = 2
                @index pxᵢ[ndim][l, I...] = x0ᵢ[ndim] + offsets[ndim][j]

                @index index[l, I...] = true
            end
        elseif N == 3
            for i in axes(offsets[1], 1), j in axes(offsets[2], 1), k in axes(offsets[3], 1)
                l = i + (j - 1) * nxdim[1] + (k - 1) * nxdim[1] * nxdim[2]
                ndim = 1
                @index pxᵢ[ndim][l, I...] = x0ᵢ[ndim] + offsets[ndim][i]
                ndim = 2
                @index pxᵢ[ndim][l, I...] = x0ᵢ[ndim] + offsets[ndim][j]
                ndim = 3
                @index pxᵢ[ndim][l, I...] = x0ᵢ[ndim] + offsets[ndim][k]

                @index index[l, I...] = true
            end
        else
            error("Unsupported number of dimensions: $N")
        end

        return nothing
    end

    @parallel (@idx nᵢ) fill_coords_index(
        pxᵢ, index, coords, nxdim, di
    )

    return Particles(backend, pxᵢ, index, nxcell, max_xcell, min_xcell, np)
end

@inline quadrant_masks(::Val{2}) = (
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1),
)

@inline quadrant_masks(::Val{3}) = (
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (1, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
)

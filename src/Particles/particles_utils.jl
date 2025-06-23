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
        backend, nxcell, max_xcell, min_xcell, xvi::Vararg{N, T}; buffer = 1 - 1.0e-5
    ) where {N, T}
    di = compute_dx(xvi)
    ni = @. length(xvi) + 1

    return init_particles(backend, nxcell, max_xcell, min_xcell, xvi, di, ni; buffer = buffer)
end

function init_particles(
        backend,
        nxcell,
        max_xcell,
        min_xcell,
        coords::NTuple{N, AbstractArray},
        dxᵢ::NTuple{N, T},
        nᵢ::NTuple{N, I};
        buffer = 1 - 1.0e-5,
    ) where {N, T, I}
    ncells = prod(nᵢ)
    np = max_xcell * ncells
    pxᵢ = ntuple(_ -> @fill(NaN, nᵢ..., celldims = (max_xcell,)), Val(N))
    index = @fill(false, nᵢ..., celldims = (max_xcell,), eltype = Bool)

    @parallel_indices (I...) function fill_coords_index(
            pxᵢ::NTuple{N, T}, index, coords, dxᵢ, nxcell, max_xcell, buffer
        ) where {N, T}

        ICA = I .+ 1 # index of the cell array

        # lower-left corner of the cell
        x0ᵢ = ntuple(Val(N)) do ndim
            coords[ndim][I[ndim]]
        end

        # fill index array
        for l in 1:max_xcell
            if l ≤ nxcell
                ntuple(Val(N)) do ndim
                    @index pxᵢ[ndim][l, ICA...] =
                        x0ᵢ[ndim] +
                        dxᵢ[ndim] * (rand() * buffer + (1 - buffer) / 2)
                end
                @index index[l, ICA...] = true

            else
                ntuple(Val(N)) do ndim
                    @index pxᵢ[ndim][l, ICA...] = NaN
                end
            end
        end
        return nothing
    end

    @parallel (@idx nᵢ .- 2) fill_coords_index(
        pxᵢ, index, coords, dxᵢ, nxcell, max_xcell, buffer
    )

    return Particles(backend, pxᵢ, index, nxcell, max_xcell, min_xcell, np)
end

## random particles initialization
"""
    init_particles(backend, nxcell, max_xcell, min_xcell, xvi...)

Initialize a `Particles` container on the grid defined by the vertex coordinates
`xvi`.

If `nxcell` is a number, particles are distributed randomly within cell
quadrants. If it is an `NTuple`, it is interpreted as the number of particles to
place regularly along each coordinate direction within every cell.

The particle vertex and center grids stored in the returned container are
extended with periodic ghost nodes. The staggered velocity grids are stored as
provided.

# Arguments
- `backend`: KernelAbstractions backend type such as `CPU`.
- `nxcell`: either the target number of particles per cell, or an `NTuple`
  describing a structured per-dimension layout.
- `max_xcell`: number of particle slots reserved per cell.
- `min_xcell`: minimum occupancy used by reinjection routines.
- `xvi`: 1D coordinate arrays describing the mesh vertices in each dimension.

# Returns
- A `Particles` object whose coordinates and occupancy arrays are ready for
  advection/interpolation routines, with `particles.xvi` and `particles.xci`
  including one periodic ghost node on each side.

# Example
```julia
xvi = LinRange(0, 1, 33), LinRange(0, 1, 33)
particles = init_particles(CPU, 24, 48, 12, xvi...)
```
"""
function init_particles(
        backend, nxcell, max_xcell, min_xcell, xi_vel::Vararg{NTuple{N2, AbstractVector}, N1}
    ) where {N1, N2}

    return init_particles(backend, nxcell, max_xcell, min_xcell, xi_vel)
end

function init_particles(
        ::Any,
        ::Number,
        ::Any,
        ::Any,
        ::Tuple{},
    )
    throw(ArgumentError("The velocity grid cannot be empty"))
end

# random distribution
function init_particles(
        backend,
        nxcell::Number,
        max_xcell,
        min_xcell,
        xi_vel_cpu::NTuple{N, NTuple{N, AbstractVector}},
    ) where {N}

    function center_coordinates(xi_vel::NTuple{3})
        xci = (
            xi_vel[2][1][2:(end - 1)],
            xi_vel[1][2][2:(end - 1)],
            xi_vel[1][3][2:(end - 1)],
        )
        return xci
    end
    function center_coordinates(xi_vel::NTuple{2})
        xci = (
            xi_vel[2][1][2:(end - 1)],
            xi_vel[1][2][2:(end - 1)],
        )
        return xci
    end

    xi_vel = ntuple(i -> TA(backend).(xi_vel_cpu[i]), Val(N))
    xci = center_coordinates(xi_vel)
    xvi = ntuple(i -> xi_vel[i][i], Val(N))
    # add ghost nodes to the center and vertex grids
    xci, xvi = add_periodic_ghost_nodes.(xci), add_periodic_ghost_nodes.(xvi)

    di_vertex = diff.(xvi)
    di_center = diff.(xci)
    di_vel = ntuple(i -> (diff.(xi_vel[i])), Val(N))
    di = (; center = TA(backend).(di_center), vertex = TA(backend).(di_vertex), velocity = di_vel)
    _di = (;
        center = map(x -> inv.(x), di.center),
        vertex = map(x -> inv.(x), di.vertex),
        velocity = map(x -> map(y -> inv.(y), x), di.velocity),
    )

    nᵢ = length.(xci)

    # number of particles per quadrant
    NQ = N == 2 ? 4 : 8
    np_quadrant = ceil(Int, nxcell / NQ)
    nxcell = np_quadrant * NQ
    max_xcell = max(nxcell, max_xcell)
    np = max_xcell * prod(nᵢ)
    # particle storage inherits the grid precision (a bare NaN would force Float64)
    T = eltype(first(xvi))
    pxᵢ = ntuple(_ -> cell_array(backend, convert(T, NaN), (max_xcell,), nᵢ), Val(N))
    index = cell_array(backend, false, (max_xcell,), nᵢ)

    launch!(
        ka_backend(index), fill_coords_index!, nᵢ .- 2,
        pxᵢ, index, xvi, di.vertex, np_quadrant
    )

    return Particles(backend, pxᵢ, index, nxcell, max_xcell, min_xcell, np, di, _di, xci, xvi, xi_vel)
end

function init_particles(
        backend,
        nxcell::Number,
        max_xcell,
        min_xcell,
        xi_vel_cpu::NTuple{N, NTuple{N, R}},
    ) where {N, R <: AbstractRange}

    function center_coordinates(xi_vel::NTuple{3})
        xci = (
            xi_vel[2][1][2:(end - 1)],
            xi_vel[1][2][2:(end - 1)],
            xi_vel[1][3][2:(end - 1)],
        )
        return xci
    end
    function center_coordinates(xi_vel::NTuple{2})
        xci = (
            xi_vel[2][1][2:(end - 1)],
            xi_vel[1][2][2:(end - 1)],
        )
        return xci
    end

    T = eltype(first(first(xi_vel_cpu)))
    xi_vel = recast_grid(xi_vel_cpu, T)
    xci = center_coordinates(xi_vel)
    xvi = ntuple(i -> xi_vel[i][i], Val(N))
    # add ghost nodes to the center and vertex grids
    xci, xvi = add_periodic_ghost_nodes.(xci), add_periodic_ghost_nodes.(xvi)

    di_vertex = getindex.(xvi, 2) .- first.(xvi)
    di_center = getindex.(xci, 2) .- first.(xci)
    di_vel = ntuple(i -> getindex.(xi_vel[i], 2) .- first.(xi_vel[i]), Val(N))
    di = (; center = di_center, vertex = di_vertex, velocity = di_vel)

    _di = (;
        center = map(x -> inv.(x), di.center),
        vertex = map(x -> inv.(x), di.vertex),
        velocity = map(x -> map(y -> inv.(y), x), di.velocity),
    )

    nᵢ = length.(xci)

    # number of particles per quadrant
    NQ = N == 2 ? 4 : 8
    np_quadrant = ceil(Int, nxcell / NQ)
    nxcell = np_quadrant * NQ
    max_xcell = max(nxcell, max_xcell)
    np = max_xcell * prod(nᵢ)
    # particle storage inherits the grid precision (a bare NaN would force Float64)
    pxᵢ = ntuple(_ -> cell_array(backend, convert(T, NaN), (max_xcell,), nᵢ), Val(N))
    index = cell_array(backend, false, (max_xcell,), nᵢ)

    launch!(
        ka_backend(index), fill_coords_index!, nᵢ .- 2,
        pxᵢ, index, xvi, di.vertex, np_quadrant
    )

    return Particles(backend, pxᵢ, index, nxcell, max_xcell, min_xcell, np, di, _di, xci, xvi, xi_vel)
end

@kernel function fill_coords_index!(
        pxᵢ::NTuple{N, T}, index, coords, di::NTuple{N}, np_quadrant
    ) where {N, T}
    I0 = @index(Global, NTuple)
    I = I0 .+ 1 # shift by one to skip the periodic ghost node
    # lower-left corner of the cell
    x0ᵢ = ntuple(Val(N)) do ndim
        @inline
        coords[ndim][I[ndim]]
    end
    dxᵢ = @dxi di I...
    # coordinate scalar type: bare 0.5/rand() literals would promote to Float64
    Tp = typeof(first(x0ᵢ))
    masks = quadrant_masks(Val(N))
    # fill index array
    l = 0 # particle counter
    for iq in eachindex(masks)
        xcᵢ = x0ᵢ .+ dxᵢ .* masks[iq] ./ 2 # quadrant lower-left coordinates
        for _ in 1:np_quadrant
            l += 1
            for ndim in 1:N
                CAI.@index pxᵢ[ndim][l, I...] = xcᵢ[ndim] + dxᵢ[ndim] / 2 * rand(Tp)
            end
            CAI.@index index[l, I...] = true
        end
    end
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
    # particle storage inherits the grid precision (a bare NaN would force Float64)
    T = eltype(first(coords))
    pxᵢ = ntuple(_ -> cell_array(backend, convert(T, NaN), (max_xcell,), nᵢ), Val(N))
    index = cell_array(backend, false, (max_xcell,), nᵢ)

    di = compute_dx(coords)

    launch!(
        ka_backend(index), fill_regular_coords_index!, nᵢ,
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

@kernel function fill_regular_coords_index!(
        pxᵢ::NTuple{N, T}, index, coords, nxdim, di
    ) where {N, T}
    I = @index(Global, NTuple)

    dxi = @dxi di I...

    # lower-left corner of the cell
    x0ᵢ = ntuple(Val(N)) do ndim
        coords[ndim][I[ndim]]
    end

    # fill index array
    local_dx = @. dxi / (nxdim + 1)
    if N == 2
        for i in 1:nxdim[1], j in 1:nxdim[2]
            l = i + (j - 1) * nxdim[1]
            ndim = 1
            CAI.@index pxᵢ[ndim][l, I...] = x0ᵢ[ndim] + l * local_dx[ndim][i]
            ndim = 2
            CAI.@index pxᵢ[ndim][l, I...] = x0ᵢ[ndim] + l * local_dx[ndim][j]

            CAI.@index index[l, I...] = true
        end
    elseif N == 3
        for i in 1:nxdim[1], j in 1:nxdim[2], k in 1:nxdim[3]
            l = i + (j - 1) * nxdim[1] + (k - 1) * nxdim[1] * nxdim[2]
            ndim = 1
            CAI.@index pxᵢ[ndim][l, I...] = x0ᵢ[ndim] + l * local_dx[ndim][i]
            ndim = 2
            CAI.@index pxᵢ[ndim][l, I...] = x0ᵢ[ndim] + l * local_dx[ndim][j]
            ndim = 3
            CAI.@index pxᵢ[ndim][l, I...] = x0ᵢ[ndim] + l * local_dx[ndim][k]

            CAI.@index index[l, I...] = true
        end
    end
end

"""
    init_cell_arrays(particles::Particles, ::Val{N})

Allocate `N` cell-aligned scratch arrays with the same cell layout as
`particles.coords`.

This is mainly used internally to create per-particle temporary storage for
quantities such as interpolated fields or time-integration work arrays.

# Returns
- An `N`-tuple of `CellArray`s with the same particle-cell layout as
  `particles.coords`.
"""
@inline function init_cell_arrays(particles::Particles{Backend}, ::Val{N}) where {Backend, N}
    # scratch arrays inherit the particle precision (Metal has no Float64)
    T = eltype(eltype(particles.coords[1]))
    return ntuple(
        _ -> cell_array(Backend, zero(T), cellsize(particles.index), size(particles.coords[1])),
        Val(N),
    )
end

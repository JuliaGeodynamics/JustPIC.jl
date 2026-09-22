## particles initialization
"""
    init_particles(backend, nxcell, max_xcell, min_xcell, grid_vx, grid_vy[, grid_vz])

Initialize a `Particles` container from the staggered velocity grids.

Each velocity component is supplied as an `N`-tuple of coordinate vectors. The
diagonal coordinate vector of each component defines the particle vertex grid;
the off-diagonal vectors define the cell-center grid. For example, in 2D pass
`grid_vx = (xv, yc_extended)` and `grid_vy = (xc_extended, yv)`.

If `nxcell` is a number, particles are distributed randomly within cell
quadrants; the count is rounded up to a multiple of the number of quadrants so
that every quadrant holds the same number of particles. If it is an `NTuple`, it
gives the number of particles placed along each coordinate direction of every
cell: `nxcell[d]` particles sit at the centers of a uniform sub-grid of spacing
`dx[d] / nxcell[d]`, so no particle lands on a cell boundary and the spacing is
uniform across the whole domain when the grid is.

In both cases `max_xcell` is raised to the resulting number of particles per
cell if it is smaller.

The particle vertex and center grids stored in the returned container are
extended with periodic ghost nodes. The staggered velocity grids are stored as
provided.

# Arguments
- `backend`: KernelAbstractions backend type such as `CPU`.
- `nxcell`: either the target number of particles per cell, or an `NTuple`
  describing a structured per-dimension layout.
- `max_xcell`: number of particle slots reserved per cell.
- `min_xcell`: minimum occupancy used by reinjection routines.
- `grid_vx`, `grid_vy`, `grid_vz`: staggered velocity-grid coordinate tuples.
  Omit `grid_vz` for a 2D simulation. Each tuple must contain one coordinate
  vector per spatial dimension.

# Returns
- A `Particles` object whose coordinates and occupancy arrays are ready for
  advection/interpolation routines, with `particles.xvi` and `particles.xci`
  including one periodic ghost node on each side.

# Example
```julia
xv, yv = LinRange(0, 1, 33), LinRange(0, 1, 33)
dx = xv[2] - xv[1]
xc = LinRange(dx / 2, 1 - dx / 2, 32)
yc = xc
grid_vx = xv, LinRange(first(yc) - dx, last(yc) + dx, 34)
grid_vy = LinRange(first(xc) - dx, last(xc) + dx, 34), yv
particles = init_particles(CPU, 24, 48, 12, grid_vx, grid_vy)

# 5x5 regularly spaced particles per cell
particles = init_particles(CPU, (5, 5), 48, 12, grid_vx, grid_vy)
```
"""
function init_particles(
        backend, nxcell, max_xcell, min_xcell, xi_vel::Vararg{NTuple{N2, AbstractVector}, N1}
    ) where {N1, N2}

    return init_particles(backend, nxcell, max_xcell, min_xcell, xi_vel)
end

@noinline _throw_empty_velocity_grid() = throw(ArgumentError("The velocity grid cannot be empty"))

init_particles(::Any, ::Any, ::Any, ::Any, ::Tuple{}) = _throw_empty_velocity_grid()

# `Tuple{}` is also `NTuple{0, NTuple{0, AbstractVector}}`, which makes the
# velocity-grid methods below applicable; this signature is more specific than
# all of them and keeps the empty grid an error instead of an ambiguity.
init_particles(::Any, ::Union{Number, Tuple{}}, ::Any, ::Any, ::Tuple{}) = _throw_empty_velocity_grid()

function init_particles(
        backend,
        nxcell::Union{Number, NTuple{N, Integer}},
        max_xcell,
        min_xcell,
        xi_vel_cpu::NTuple{N, NTuple{N, AbstractVector}},
    ) where {N}

    xi_vel, xci, xvi, di, _di = staggered_grids(backend, xi_vel_cpu)
    return _init_particles(backend, nxcell, max_xcell, min_xcell, xi_vel, xci, xvi, di, _di)
end

function init_particles(
        backend,
        nxcell::Union{Number, NTuple{N, Integer}},
        max_xcell,
        min_xcell,
        xi_vel_cpu::NTuple{N, NTuple{N, R}},
    ) where {N, R <: AbstractRange}

    xi_vel, xci, xvi, di, _di = staggered_grids(backend, xi_vel_cpu)
    return _init_particles(backend, nxcell, max_xcell, min_xcell, xi_vel, xci, xvi, di, _di)
end

# The diagonal coordinate vector of each velocity component is the vertex grid;
# the off-diagonal ones are the cell-center grid.
@inline center_coordinates(xi_vel::NTuple{2}) = (
    xi_vel[2][1][2:(end - 1)],
    xi_vel[1][2][2:(end - 1)],
)

@inline center_coordinates(xi_vel::NTuple{3}) = (
    xi_vel[2][1][2:(end - 1)],
    xi_vel[1][2][2:(end - 1)],
    xi_vel[1][3][2:(end - 1)],
)

@inline function inverse_spacing(di)
    return (;
        center = map(x -> inv.(x), di.center),
        vertex = map(x -> inv.(x), di.vertex),
        velocity = map(x -> map(y -> inv.(y), x), di.velocity),
    )
end

"""
    staggered_grids(backend, xi_vel_cpu)

Build the device-resident grids carried by a [`Particles`](@ref) container from
the staggered velocity grids: the velocity grids themselves, the cell-center and
vertex grids extended with one periodic ghost node on each side, and the cell
spacings together with their reciprocals.
"""
function staggered_grids(backend, xi_vel_cpu::NTuple{N, NTuple{N, AbstractVector}}) where {N}
    xci_cpu = center_coordinates(xi_vel_cpu)
    xvi_cpu = ntuple(i -> xi_vel_cpu[i][i], Val(N))
    xi_vel = ntuple(i -> TA(backend).(xi_vel_cpu[i]), Val(N))
    xci = TA(backend).(add_periodic_ghost_nodes.(xci_cpu))
    xvi = TA(backend).(add_periodic_ghost_nodes.(xvi_cpu))

    di_vertex = diff.(xvi)
    di_center = diff.(xci)
    di_vel = ntuple(i -> (diff.(xi_vel[i])), Val(N))
    di = (; center = TA(backend).(di_center), vertex = TA(backend).(di_vertex), velocity = di_vel)

    return xi_vel, xci, xvi, di, inverse_spacing(di)
end

function staggered_grids(backend, xi_vel_cpu::NTuple{N, NTuple{N, R}}) where {N, R <: AbstractRange}
    T = eltype(first(first(xi_vel_cpu)))
    xi_vel = recast_grid(xi_vel_cpu, T)
    xci = center_coordinates(xi_vel)
    xvi = ntuple(i -> xi_vel[i][i], Val(N))
    # add ghost nodes to the center and vertex grids
    xci = recast_grid(add_periodic_ghost_nodes.(xci), T)
    xvi = recast_grid(add_periodic_ghost_nodes.(xvi), T)

    di_vertex = getindex.(xvi, 2) .- first.(xvi)
    di_center = getindex.(xci, 2) .- first.(xci)
    di_vel = ntuple(i -> getindex.(xi_vel[i], 2) .- first.(xi_vel[i]), Val(N))
    di = (; center = di_center, vertex = di_vertex, velocity = di_vel)

    return xi_vel, xci, xvi, di, inverse_spacing(di)
end

# random distribution, balanced over the cell quadrants
function _init_particles(
        backend, nxcell::Number, max_xcell, min_xcell, xi_vel, xci, xvi::NTuple{N}, di, _di
    ) where {N}

    nᵢ = length.(xci)

    # number of particles per quadrant
    NQ = N == 2 ? 4 : 8
    np_quadrant = ceil(Int, nxcell / NQ)
    nxcell = np_quadrant * NQ
    max_xcell = max(nxcell, max_xcell)
    np = max_xcell * prod(nᵢ)
    pxᵢ, index = allocate_particle_storage(backend, xvi, max_xcell, nᵢ)

    launch!(
        ka_backend(index), fill_coords_index!, nᵢ .- 2,
        pxᵢ, index, xvi, di.vertex, np_quadrant
    )

    return Particles(backend, pxᵢ, index, nxcell, max_xcell, min_xcell, np, di, _di, xci, xvi, xi_vel)
end

# regular distribution, `nxdim[d]` particles along dimension `d` of every cell
function _init_particles(
        backend, nxdim::NTuple{N, Integer}, max_xcell, min_xcell, xi_vel, xci, xvi::NTuple{N}, di, _di
    ) where {N}

    all(>(0), nxdim) || throw(ArgumentError("The number of particles per cell direction must be positive, got $nxdim"))

    nᵢ = length.(xci)

    nxcell = prod(nxdim)
    max_xcell = max(nxcell, max_xcell)
    np = max_xcell * prod(nᵢ)
    pxᵢ, index = allocate_particle_storage(backend, xvi, max_xcell, nᵢ)

    launch!(
        ka_backend(index), fill_regular_coords_index!, nᵢ .- 2,
        pxᵢ, index, xvi, di.vertex, nxdim
    )

    return Particles(backend, pxᵢ, index, nxcell, max_xcell, min_xcell, np, di, _di, xci, xvi, xi_vel)
end

function allocate_particle_storage(backend, xvi::NTuple{N}, max_xcell, nᵢ) where {N}
    # particle storage inherits the grid precision (a bare NaN would force Float64)
    T = eltype(first(xvi))
    pxᵢ = ntuple(_ -> cell_array(backend, convert(T, NaN), (max_xcell,), nᵢ), Val(N))
    index = cell_array(backend, false, (max_xcell,), nᵢ)
    return pxᵢ, index
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
        pxᵢ::NTuple{N, T}, index, coords, di::NTuple{N}, nxdim::NTuple{N}
    ) where {N, T}
    I0 = @index(Global, NTuple)
    I = I0 .+ 1 # shift by one to skip the periodic ghost node
    # lower-left corner of the cell
    x0ᵢ = ntuple(Val(N)) do ndim
        @inline
        coords[ndim][I[ndim]]
    end
    dxᵢ = @dxi di I...
    # particles sit at the centers of a uniform sub-grid of the cell: no particle
    # lands on a cell boundary, and the spacing stays uniform across boundaries
    local_dx = dxᵢ ./ nxdim

    # `l` counts particle slots within the cell, so a linear counter is wanted here
    for (l, sub) in enumerate(CartesianIndices(nxdim))
        for ndim in 1:N
            CAI.@index pxᵢ[ndim][l, I...] = x0ᵢ[ndim] + (sub[ndim] - 1) * local_dx[ndim] + local_dx[ndim] / 2
        end
        CAI.@index index[l, I...] = true
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

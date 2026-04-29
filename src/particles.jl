abstract type AbstractParticles end

"""
    Particles{Backend, N, I, T1, T2, D, V} <: AbstractParticles

Main particle container used by JustPIC for material points stored cell-by-cell
in `CellArray`s.

`coords` is an `N`-tuple of particle-coordinate arrays, `index` marks which
slots are active inside each cell, `nxcell` is the target initial occupancy per
cell, and `min_xcell`/`max_xcell` define the occupancy range used by injection
and cleanup routines.

Use `init_particles` to construct this type instead of calling the inner
constructor directly.
"""
struct Particles{Backend, N, I, T1, T2, D, V} <: AbstractParticles
    coords::NTuple{N, T1}              # particle coordinates
    index::T2                          # BitArray (true if particle in that memory space)
    nxcell::I                          # initial particles per cell
    max_xcell::I                       # max particles per cell
    min_xcell::I                       # min particles per cell
    np::I                              # total number of particles
    di::D                              # grid spacing
    _di::D                             # inverse grid spacing
    xci::NTuple{N, V}                  # cell-centered grid
    xvi::NTuple{N, V}                  # vertex-centered grid
    xi_vel::NTuple{N, NTuple{N, V}} # velocity grid

    function Particles(
            ::Type{B},
            coords::NTuple{N, T1},
            index::T2,
            nxcell::I,
            max_xcell::I,
            min_xcell::I,
            np::I,
            di::D,
            _di::D,
            xci::NTuple{N, V},
            xvi::NTuple{N, V},
            xi_vel::NTuple{N, NTuple{N, V}},
        ) where {B, N, I, T1, T2, D, V}
        return new{B, N, I, T1, T2, D, V}(coords, index, nxcell, max_xcell, min_xcell, np, di, _di, xci, xvi, xi_vel)
    end
end

function Particles(coords, index::CPUCellArray, nxcell, max_xcell, min_xcell, np, di, _di, xci, xvi, xi_vel)
    return Particles(CPUBackend, coords, index, nxcell, max_xcell, min_xcell, np, di, _di, xci, xvi, xi_vel)
end

"""
    MarkerChain{Backend,N,I,T1,T2,T3,TV} <: AbstractParticles

Container for a 2D marker chain used to represent a free surface or topographic
interface.

The chain stores current and previous marker coordinates, the topography sampled
at grid vertices, and a per-cell activity mask.

Use `init_markerchain` to create a chain, `fill_chain_from_chain!` or
`fill_chain_from_vertices!` to overwrite its geometry, and
`advect_markerchain!` to evolve it in time.
"""
struct MarkerChain{Backend, N, I, T1, T2, T3, TV} <: AbstractParticles
    coords::NTuple{N, T1}    # current x-coord in 2D, (x,y)
    coords0::NTuple{N, T1}   # x-coord in 2D, (x,y) from the previous time step
    h_vertices::T2          # topography at the vertices of the grid (current)
    h_vertices0::T2         # topography at the vertices of the grid (previous timestep)
    cell_vertices::TV
    index::T3
    max_xcell::I
    min_xcell::I

    function MarkerChain(
            ::Type{B},
            coords::NTuple{N, T1},
            coords0::NTuple{N, T1},
            h_vertices::T2,
            h_vertices0::T2,
            cell_vertices::TV,
            index::T3,
            min_xcell::I,
            max_xcell::I,
        ) where {B, N, I, T1, T2, T3, TV}
        return new{B, N, I, T1, T2, T3, TV}(
            coords,
            coords0,
            h_vertices,
            h_vertices0,
            cell_vertices,
            index,
            max_xcell,
            min_xcell,
        )
    end
end

function MarkerChain(coords, index::CPUCellArray, cell_vertices, min_xcell, max_xcell)
    return MarkerChain(
        CPUBackend,
        coords,
        coords0,
        h_vertices,
        h_vertices0,
        cell_vertices,
        index,
        max_xcell,
        min_xcell,
    )
end

"""
    MarkerSurface{Backend, I, T2, TV, TW} <: AbstractParticles

A 3D free surface tracker using a structured marker grid.
The surface is represented as a 2D grid of topography values (z-heights) at corner nodes.

# Fields
- `topo::T2`       — topography (z-elevation) at grid vertices, size `(nx+1, ny+1)`
- `topo0::T2`      — topography from the previous time step
- `vx::T2`         — x-velocity interpolated to surface nodes
- `vy::T2`         — y-velocity interpolated to surface nodes
- `vz::T2`         — z-velocity interpolated to surface nodes
- `xv::TV`         — x-coordinates of surface grid vertices
- `yv::TV`         — y-coordinates of surface grid vertices
- `air_phase::I`   — phase ID of the air/sticky-air layer
- `workspace::TW`  — pre-allocated workspace buffers for allocation-free timesteps
"""
struct MarkerSurface{Backend, I, T2, TV, TW} <: AbstractParticles
    topo::T2             # topography at grid vertices (nx+1) x (ny+1)
    topo0::T2            # previous-timestep topography
    vx::T2               # surface velocity x-component at vertices
    vy::T2               # surface velocity y-component at vertices
    vz::T2               # surface velocity z-component at vertices
    xv::TV               # x vertex coordinates
    yv::TV               # y vertex coordinates
    air_phase::I         # sticky-air phase ID
    workspace::TW        # pre-allocated workspace buffers

    function MarkerSurface(
            ::Type{B},
            topo::T2, topo0::T2,
            vx::T2, vy::T2, vz::T2,
            xv::TV, yv::TV,
            air_phase::I,
            workspace::TW,
        ) where {B, I, T2, TV, TW}
        return new{B, I, T2, TV, TW}(
            topo, topo0, vx, vy, vz, xv, yv,
            air_phase, workspace,
        )
    end
end

function MarkerSurface(
        topo, topo0, vx, vy, vz, xv, yv, air_phase, workspace
    )
    return MarkerSurface(
        CPUBackend,
        topo, topo0, vx, vy, vz,
        xv, yv,
        air_phase,
        workspace,
    )
end

"""
    PassiveMarkers{Backend,T} <: AbstractParticles

Lightweight particle container for passive tracers that only store coordinates.

Unlike `Particles`, passive markers do not keep per-cell occupancy metadata and
are intended for tracer-style advection and interpolation workflows where the
markers do not feed back into the simulation.

Use `init_passive_markers` to construct this type.
"""
struct PassiveMarkers{Backend, T} <: AbstractParticles
    coords::T
    np::Int64

    function PassiveMarkers(::Type{B}, coords::NTuple) where {B}
        # np = length(coords[1].data)
        np = length(coords[1])
        return new{B, typeof(coords)}(coords, np)
    end
    function PassiveMarkers(::Type{B}, coords::AbstractArray) where {B}
        np = length(coords)
        return new{B, typeof(coords)}(coords, np)
    end
end

function PassiveMarkers(coords::Union{AbstractArray, NTuple{N, T}}) where {N, T}
    return PassiveMarkers(CPUBackend, coords)
end

# useful functions

unwrap_abstractarray(x::AbstractArray) = typeof(x).name.wrapper

@inline count_particles(p::AbstractParticles, icell::Vararg{Int, N}) where {N} =
    count(p.index[icell...])

"""
    cell_length(chain::MarkerChain)

Return the horizontal cell size of a 2D marker chain.

This is the spacing between consecutive entries in `chain.cell_vertices`.
"""
@inline cell_length(p::MarkerChain{B, 2}) where {B} = p.cell_vertices[2] - p.cell_vertices[1]
@inline cell_length_x(p::MarkerChain{B, 3}) where {B} =
    p.cell_vertices[1][2] - p.cell_vertices[1][1]
@inline cell_length_y(p::MarkerChain{B, 3}) where {B} =
    p.cell_vertices[2][2] - p.cell_vertices[2][1]

@inline cell_x(p::AbstractParticles, icell::Vararg{Int, N}) where {N} = p.coords[1][icell...]
@inline cell_y(p::AbstractParticles, icell::Vararg{Int, N}) where {N} = p.coords[2][icell...]
@inline cell_z(p::AbstractParticles, icell::Vararg{Int, N}) where {N} = p.coords[3][icell...]

"""
    cell_index(x, dx)
    cell_index(x, xv)
    cell_index(x, xv, dx)

Return the one-based parent-cell index containing coordinate `x`.

The overloads support regular grids defined by a scalar spacing `dx`, as well as
shifted or nonzero-origin grids defined by a coordinate vector `xv`.

For tuple-valued coordinates, the tuple overload returns one cell index per
dimension.
"""
# @inline cell_index(xᵢ::T, dxᵢ::T) where {T} = abs(Int(xᵢ ÷ dxᵢ)) + 1
@inline cell_index(xᵢ::T, dxᵢ::T) where {T} = abs(Int(trunc(xᵢ / dxᵢ))) + 1
@inline cell_index(xᵢ::T, xvᵢ::AbstractVector{T}) where {T} =
    cell_index(xᵢ, xvᵢ, abs(xvᵢ[2] - xvᵢ[1]))

# generic one that works for any grid
@inline function cell_index(xᵢ::T, xvᵢ::AbstractVector{T}, dxᵢ::T) where {T}
    xv₀ = first(xvᵢ)
    if iszero(xv₀)
        return cell_index(xᵢ, dxᵢ)
    else
        return cell_index(xᵢ - xv₀, dxᵢ)
    end
end

@inline function cell_index(x::NTuple{N, T}, xv::NTuple{N, AbstractVector{T}}) where {N, T}
    return ntuple(Val(N)) do i
        Base.@_inline_meta
        cell_index(x[i], xv[i])
    end
end

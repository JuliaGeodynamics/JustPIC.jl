"""
    SubgridDiffusionCellArrays(particles; loc = :vertex)

Allocate scratch storage used by the subgrid thermal diffusion routines.

The returned object stores old particle temperatures, per-particle temperature
increments, characteristic diffusion timescales, and a grid-sized accumulation
buffer.

`loc` selects whether the accumulation buffer should match a vertex-based or
cell-centered grid layout.
"""
struct SubgridDiffusionCellArrays{CA, T}
    pT0::CA # particle old temperature
    pΔT::CA # particle temperature increment
    dt₀::CA # characteristic timescale `dt₀` of the local cell := ρCp / (K * (2/Δx^2 + 2/Δy^2))
    ΔT_subgrid::T # subgrid temperature increment

    function SubgridDiffusionCellArrays(particles::Particles{Backend}; loc::Symbol = :vertex) where {Backend}
        pΔT, pT0, dt₀ = init_cell_arrays(particles, Val(3))
        ni = if loc === :vertex
            size(pΔT) .+ 1
        elseif loc === :center
            size(pΔT)
        end
        ΔT = TA(Backend)(zeros(ni...))
        CA = typeof(pΔT)
        T = typeof(ΔT)
        return new{CA, T}(pT0, pΔT, dt₀, ΔT)
    end
end

function SubgridDiffusionCellArrays(::T) where {T}
    throw(ArgumentError("SubgridDiffusionCellArrays: $T backend not supported"))
end

"""
    subgrid_diffusion!(pT, T_grid, ΔT_grid, subgrid_arrays, particles, dt; d = 1.0)

Apply the vertex-based subgrid diffusion correction to particle temperatures.

Temperatures are interpolated from the grid to particles, relaxed using the local
subgrid model, mapped back to the grid as a correction, and then reapplied to the
particle temperatures.

# Arguments
- `pT`: particle temperature field updated in place.
- `T_grid`: source temperature on the vertex grid.
- `ΔT_grid`: resolved-grid temperature increment.
- `subgrid_arrays`: scratch storage created with `SubgridDiffusionCellArrays`.
- `particles`: particle container.
- `dt`: timestep.
- `d`: dimensionless subgrid diffusion coefficient.
"""
function subgrid_diffusion!(
        pT, T_grid, ΔT_grid, subgrid_arrays, particles::Particles, dt; d = 1.0
    )
    # d = dimensionless numerical diffusion coefficient (0 ≤ d ≤ 1)
    (; pT0, pΔT, dt₀) = subgrid_arrays
    ni = size(pT)

    launch!(ka_backend(pT), memcopy_cellarray_kernel!, ni, pT0, pT)
    grid2particle!(pT, T_grid, particles)

    launch!(ka_backend(pT), subgrid_diffusion_kernel!, ni, pT, pT0, pΔT, dt₀, particles.index, d, dt)
    particle2grid!(subgrid_arrays.ΔT_subgrid, pΔT, particles)

    launch!(ka_backend(subgrid_arrays.ΔT_subgrid), update_ΔT_subgrid_kernel!, ni .+ 1, subgrid_arrays.ΔT_subgrid, ΔT_grid)
    grid2particle!(pΔT, subgrid_arrays.ΔT_subgrid, particles)

    launch!(ka_backend(pT), update_particle_temperature_kernel!, ni, pT, pT0, pΔT)

    return nothing
end

"""
    subgrid_diffusion_centroid!(pT, T_grid, ΔT_grid, subgrid_arrays, particles, dt; d = 1.0)

Centroid-grid variant of `subgrid_diffusion!`.

Use this when the resolved temperature field lives at cell centers instead of
vertices.
"""
function subgrid_diffusion_centroid!(
        pT, T_grid, ΔT_grid, subgrid_arrays, particles, dt; d = 1.0
    )
    # d = dimensionless numerical diffusion coefficient (0 ≤ d ≤ 1)
    (; pT0, pΔT, dt₀) = subgrid_arrays
    ni = size(pT)

    launch!(ka_backend(pT), memcopy_cellarray_kernel!, ni, pT0, pT)
    centroid2particle!(pT, T_grid, particles)

    launch!(ka_backend(pT), subgrid_diffusion_kernel!, ni, pT, pT0, pΔT, dt₀, particles.index, d, dt)
    particle2centroid!(subgrid_arrays.ΔT_subgrid, pΔT, particles)

    launch!(ka_backend(subgrid_arrays.ΔT_subgrid), update_ΔT_subgrid_kernel!, ni, subgrid_arrays.ΔT_subgrid, ΔT_grid)
    centroid2particle!(pΔT, subgrid_arrays.ΔT_subgrid, particles)

    launch!(ka_backend(pT), update_particle_temperature_kernel!, ni, pT, pT0, pΔT)

    return nothing
end

@kernel function memcopy_cellarray_kernel!(A, B)
    I = @index(Global, NTuple)
    for ip in cellaxes(A)
        CAI.@index A[ip, I...] = CAI.@index(B[ip, I...])
    end
end

@kernel function subgrid_diffusion_kernel!(pT, pT0, pΔT, dt₀, index, d, dt)
    I = @index(Global, NTuple)
    for ip in cellaxes(pT)
        # early escape if there is no particle in this memory locations
        doskip(index, ip, I...) && continue

        pT0ᵢ = CAI.@index pT0[ip, I...]
        pTᵢ = CAI.@index pT[ip, I...]

        # subgrid diffusion of the i-th particle
        pΔTᵢ = (pTᵢ - pT0ᵢ) * (1 - exp(-d * dt / max(CAI.@index(dt₀[ip, I...]), 1.0e-9)))
        CAI.@index pT0[ip, I...] = pT0ᵢ + pΔTᵢ
        CAI.@index pΔT[ip, I...] = pΔTᵢ
    end
end

@kernel function update_ΔT_subgrid_kernel!(ΔTsubgrid, ΔT)
    I = @index(Global, NTuple)
    ΔTsubgrid[I...] = ΔT[I .+ 1...] - ΔTsubgrid[I...]
end

@kernel function update_particle_temperature_kernel!(pT, pT0, pΔT)
    I = @index(Global, NTuple)
    for ip in cellaxes(pT)
        CAI.@index pT[ip, I...] = CAI.@index(pT0[ip, I...]) + CAI.@index(pΔT[ip, I...])
    end
end

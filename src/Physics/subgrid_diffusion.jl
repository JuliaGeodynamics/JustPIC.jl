struct SubgridDiffusionCellArrays{CA, T}
    pT0::CA # particle old temperature
    pΔT::CA # particle temperature increment
    dt₀::CA # characteristic timescale `dt₀` of the local cell := ρCp / (K * (2/Δx^2 + 2/Δy^2))
    ΔT_subgrid::T # subgrid temperature increment

    function SubgridDiffusionCellArrays(particles::Particles; loc::Symbol = :vertex)
        pΔT, pT0, dt₀ = init_cell_arrays(particles, Val(3))
        ni = if loc === :vertex
            size(pΔT) .+ 1
        elseif loc === :center
            size(pΔT)
        end
        ΔT = @zeros(ni...)
        CA = typeof(pΔT)
        T = typeof(ΔT)
        return new{CA, T}(pT0, pΔT, dt₀, ΔT)
    end
end

function SubgridDiffusionCellArrays(::T) where {T}
    throw(ArgumentError("SubgridDiffusionCellArrays: $T backend not supported"))
end

function subgrid_diffusion!(
        pT, T_grid, ΔT_grid, subgrid_arrays, particles::Particles, xvi, di, dt; d = 1.0
    )
    # d = dimensionless numerical diffusion coefficient (0 ≤ d ≤ 1)
    (; pT0, pΔT, dt₀) = subgrid_arrays
    ni = size(pT)

    @parallel (@idx ni) memcopy_cellarray!(pT0, pT)
    grid2particle!(pT, xvi, T_grid, particles)

    @parallel (@idx ni) subgrid_diffusion!(pT, pT0, pΔT, dt₀, particles.index, d, di, dt)
    particle2grid!(subgrid_arrays.ΔT_subgrid, pΔT, xvi, particles)

    @parallel (@idx ni .+ 1) update_ΔT_subgrid!(subgrid_arrays.ΔT_subgrid, ΔT_grid)
    grid2particle!(pΔT, xvi, subgrid_arrays.ΔT_subgrid, particles)

    @parallel (@idx ni) update_particle_temperature!(pT, pT0, pΔT)

    return nothing
end

function subgrid_diffusion_centroid!(
        pT, T_grid, ΔT_grid, subgrid_arrays, particles, xci, di, dt; d = 1.0
    )
    # d = dimensionless numerical diffusion coefficient (0 ≤ d ≤ 1)
    (; pT0, pΔT, dt₀) = subgrid_arrays
    ni = size(pT)

    @parallel memcopy_cellarray!(pT0, pT)
    centroid2particle!(pT, xci, T_grid, particles)

    @parallel (@idx ni) subgrid_diffusion!(pT, pT0, pΔT, dt₀, particles.index, d, di, dt)
    particle2centroid!(subgrid_arrays.ΔT_subgrid, pΔT, xci, particles)

    @parallel (@idx ni) update_ΔT_subgrid!(subgrid_arrays.ΔT_subgrid, ΔT_grid)
    centroid2particle!(pΔT, xci, subgrid_arrays.ΔT_subgrid, particles)

    @parallel (@idx ni) update_particle_temperature!(pT, pT0, pΔT)

    return nothing
end

@parallel_indices (I...) function memcopy_cellarray!(A, B)
    for ip in cellaxes(A)
        @index A[ip, I...] = @index(B[ip, I...])
    end
    return nothing
end

@parallel_indices (I...) function subgrid_diffusion!(pT, pT0, pΔT, dt₀, index, d, di, dt)
    for ip in cellaxes(pT)
        # early escape if there is no particle in this memory locations
        doskip(index, ip, I...) && continue

        pT0ᵢ = @index pT0[ip, I...]
        pTᵢ = @index pT[ip, I...]

        # subgrid diffusion of the i-th particle
        pΔTᵢ = (pTᵢ - pT0ᵢ) * (1 - exp(-d * dt / max(@index(dt₀[ip, I...]), 1.0e-9)))
        @index pT0[ip, I...] = pT0ᵢ + pΔTᵢ
        @index pΔT[ip, I...] = pΔTᵢ
    end

    return nothing
end

@parallel_indices (I...) function update_ΔT_subgrid!(ΔTsubgrid, ΔT)
    ΔTsubgrid[I...] = ΔT[I...] - ΔTsubgrid[I...]
    return nothing
end

@parallel_indices (I...) function update_particle_temperature!(pT, pT0, pΔT)
    for ip in cellaxes(pT)
        @index pT[ip, I...] = @index(pT0[ip, I...]) + @index(pΔT[ip, I...])
    end
    return nothing
end


for moduleᵢ in JP_module, ext_backendᵢ in ext_backend, TArrayᵢ in TArray

    @eval begin
        function $(TArrayᵢ)(particles::JustPIC.Particles{JustPIC.CPUBackend}) 
            (; coords, index, nxcell, max_xcell, min_xcell, np) = particles
            coords_gpu = $(TArrayᵢ).(coords);
            return Particles($(ext_backendᵢ), coords_gpu, $(TArrayᵢ)(index), nxcell, max_xcell, min_xcell, np)
        end

        function $(TArrayᵢ)(phase_ratios::JustPIC.PhaseRatios{JustPIC.CPUBackend}) 
            (; vertex, center) = phase_ratios
            return JustPIC.PhaseRatios($(ext_backendᵢ), $(TArrayᵢ)(vertex), $(TArrayᵢ)(center))
        end

        function $(moduleᵢ).Particles(
            coords,
            index::CellArray{StaticArraysCore.SVector{N1,Bool},2,0,$(TArrayᵢ){Bool,N2}},
            nxcell,
            max_xcell,
            min_xcell,
            np,
        ) where {N1,N2}
            return Particles($(ext_backendᵢ), coords, index, nxcell, max_xcell, min_xcell, np)
        end

        function $(moduleᵢ).SubgridDiffusionCellArrays(particles::Particles{$(ext_backendᵢ)})
            return SubgridDiffusionCellArrays(particles)
        end

        function $(moduleᵢ).init_particles(
            ::Type{$(ext_backendᵢ)}, nxcell, max_xcell, min_xcell, x, y
        )
            return init_particles($(ext_backendᵢ), nxcell, max_xcell, min_xcell, x, y)
        end

        function $(moduleᵢ).init_particles(
            ::Type{$(ext_backendᵢ)},
            nxcell,
            max_xcell,
            min_xcell,
            coords::NTuple{2,AbstractArray},
            dxᵢ::NTuple{2,T},
            nᵢ::NTuple{2,I},
        ) where {T,I}
            return init_particles($(ext_backendᵢ), nxcell, max_xcell, min_xcell, coords, dxᵢ, nᵢ)
        end

        function $(moduleᵢ).advection!(
            particles::Particles{$(ext_backendᵢ)},
            method::AbstractAdvectionIntegrator,
            V,
            grid_vxi::NTuple{N,NTuple{N,T}},
            dt,
        ) where {N,T}
            return advection!(particles, method, V, grid_vxi, dt)
        end

        function $(moduleᵢ).advection_LinP!(
            particles::Particles{$(ext_backendᵢ)},
            method::AbstractAdvectionIntegrator,
            V,
            grid_vxi::NTuple{N,NTuple{N,T}},
            dt,
        ) where {N,T}
            return advection_LinP!(particles, method, V, grid_vxi, dt)
        end

        function $(moduleᵢ).advection_MQS!(
            particles::Particles{$(ext_backendᵢ)},
            method::AbstractAdvectionIntegrator,
            V,
            grid_vxi::NTuple{N,NTuple{N,T}},
            dt,
        ) where {N,T}
            return advection_MQS!(particles, method, V, grid_vxi, dt)
        end

        function $(moduleᵢ).centroid2particle!(
            Fp, xci, F::$(TArrayᵢ), particles::Particles{$(ext_backendᵢ)}
        )
            return centroid2particle!(Fp, xci, F, particles)
        end

        function $(moduleᵢ).grid2particle!(
            Fp, xvi, F::$(TArrayᵢ), particles::Particles{$(ext_backendᵢ)}
        )
            return grid2particle!(Fp, xvi, F, particles)
        end

        function $(moduleᵢ).particle2grid_centroid!(
            F::$(TArrayᵢ), Fp, xi::NTuple, particles::Particles{$(ext_backendᵢ)}
        )
            return particle2grid_centroid!(F, Fp, xi, particles)
        end

        function $(moduleᵢ).particle2grid!(F::$(TArrayᵢ), Fp, xi, particles)
            return particle2grid!(F, Fp, xi, particles)
        end

        function $(moduleᵢ).grid2particle_flip!(Fp, xvi, F::$(TArrayᵢ), F0, particles; α=0.0)
            return grid2particle_flip!(Fp, xvi, F, F0, particles; α=α)
        end

        function $(moduleᵢ).inject_particles!(particles::Particles{$(ext_backendᵢ)}, args, grid)
            return inject_particles!(particles, args, grid)
        end

        function $(moduleᵢ).inject_particles_phase!(
            particles::Particles{$(ext_backendᵢ)}, particles_phases, args, fields, grid
        )
            inject_particles_phase!(particles::Particles, particles_phases, args, fields, grid)
            return nothing
        end

        function $(moduleᵢ).move_particles!(
            particles::Particles{$(ext_backendᵢ)}, grid::NTuple{N}, args
        ) where {N}
            return move_particles!(particles, grid, args)
        end

        function $(moduleᵢ).init_cell_arrays(
            particles::Particles{$(ext_backendᵢ)}, V::Val{N}
        ) where {N}
            return init_cell_arrays(particles, V)
        end

        function $(moduleᵢ).subgrid_diffusion!(
            pT,
            T_grid,
            ΔT_grid,
            subgrid_arrays,
            particles::Particles{$(ext_backendᵢ)},
            xvi,
            di,
            dt;
            d=1.0,
        )
            subgrid_diffusion!(pT, T_grid, ΔT_grid, subgrid_arrays, particles, xvi, di, dt; d=d)
            return nothing
        end

        ## MakerChain

        function $(moduleᵢ).advect_markerchain!(
            chain::MarkerChain{$(ext_backendᵢ)},
            method::AbstractAdvectionIntegrator,
            V,
            grid_vxi,
            dt,
        )
            return advect_markerchain!(chain, method, V, grid_vxi, dt)
        end

        ## PassiveMarkers

        function $(moduleᵢ).init_passive_markers(
            ::Type{$(ext_backendᵢ)}, coords::NTuple{N,$(TArrayᵢ)}
        ) where {N}
            return init_passive_markers($(ext_backendᵢ), coords)
        end

        function $(moduleᵢ).advection!(
            particles::PassiveMarkers{$(ext_backendᵢ)},
            method::AbstractAdvectionIntegrator,
            V::NTuple{N,$(TArrayᵢ)},
            grid_vxi,
            dt,
        ) where {N}
            return advection!(particles, method, V, grid_vxi, dt)
        end

        function $(moduleᵢ).grid2particle!(Fp, xvi, F, particles::PassiveMarkers{$(ext_backendᵢ)})
            grid2particle!(Fp, xvi, F, particles)
            return nothing
        end

        function $(moduleᵢ).grid2particle!(
            Fp::NTuple{N,$(TArrayᵢ)},
            xvi,
            F::NTuple{N,$(TArrayᵢ)},
            particles::PassiveMarkers{$(ext_backendᵢ)},
        ) where {N}
            grid2particle!(Fp, xvi, F, particles)
            return nothing
        end

        function $(moduleᵢ).particle2grid!(
            F, Fp, buffer, xi, particles::PassiveMarkers{$(ext_backendᵢ)}
        )
            particle2grid!(F, Fp, buffer, xi, particles)
            return nothing
        end

        # Phase ratio kernels

        function $(moduleᵢ).update_phase_ratios!(phase_ratios::JustPIC.PhaseRatios{$(ext_backendᵢ)}, particles, xci, xvi, phases)
            phase_ratios_center!(phase_ratios, particles, xci, phases)
            phase_ratios_vertex!(phase_ratios, particles, xvi, phases)
            return nothing
        end

        function $(moduleᵢ).PhaseRatios(
            ::Type{$(ext_backendᵢ)}, nphases::Integer, ni::NTuple{N,Integer}
        ) where {N}
            return $(moduleᵢ).PhaseRatios(Float64, $(ext_backendᵢ), nphases, ni)
        end

        function $(moduleᵢ).PhaseRatios(
            ::Type{T}, ::Type{$(ext_backendᵢ)}, nphases::Integer, ni::NTuple{N,Integer}
        ) where {N,T}
            center = cell_array(0.0, (nphases,), ni)
            vertex = cell_array(0.0, (nphases,), ni .+ 1)

            return JustPIC.PhaseRatios($(ext_backendᵢ), center, vertex)
        end

        function $(moduleᵢ).phase_ratios_center!(
            phase_ratios::JustPIC.PhaseRatios{$(ext_backendᵢ)}, particles, xci, phases
        )
            ni = size(phases)
            di = compute_dx(xci)

            @parallel (@idx ni) phase_ratios_center_kernel!(
                phase_ratios.center, particles.coords, xci, di, phases
            )
            return nothing
        end

        function $(moduleᵢ).phase_ratios_vertex!(
            phase_ratios::JustPIC.PhaseRatios{$(ext_backendᵢ)}, particles, xvi, phases
        )
            ni = size(phases) .+ 1
            di = compute_dx(xvi)

            @parallel (@idx ni) phase_ratios_vertex_kernel!(
                phase_ratios.vertex, particles.coords, xvi, di, phases
            )
            return nothing
        end
    end
end
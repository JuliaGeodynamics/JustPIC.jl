"""
    advect_markerchain!(chain, method, V, grid_vxi, dt)

Advect a marker chain for one time step and rebuild its derived topography data.

This convenience wrapper runs marker advection, reassigns markers to cells,
resamples the chain, updates vertex elevations, and enforces mean-height
conservation.

Use this when evolving a free surface or interface represented by a
`MarkerChain`.
"""
function advect_markerchain!(
        chain::MarkerChain, method::AbstractAdvectionIntegrator, V, grid_vxi, dt
    )
    advection!(chain, method, V, grid_vxi, dt)
    move_particles!(chain)
    resample!(chain)

    # interpolate from markers to grid
    compute_topography_vertex!(chain)
    # correct topo to conserve mass
    chain.h_vertices .-= mean_height(chain.h_vertices) - mean_height(chain.h_vertices0)
    # reconstruct chain from vertices
    reconstruct_chain_from_vertices!(chain)
    copyto!(chain.coords0[1].data, chain.coords[1].data)
    copyto!(chain.coords0[2].data, chain.coords[2].data)
    # update old nodal topography
    copyto!(chain.h_vertices0, chain.h_vertices)

    return nothing
end

# Two-step Runge-Kutta advection scheme for marker chains
"""
    advection!(chain::MarkerChain, method, V, grid_vi, dt)

Advect the marker coordinates in `chain` through the staggered velocity field `V`
without performing resampling or topography reconstruction.

This lower-level method is useful if you want to customize the post-advection
marker-chain processing yourself.
"""
function advection!(
        chain::MarkerChain,
        method::AbstractAdvectionIntegrator,
        V,
        grid_vi::NTuple{N, NTuple{N, T}},
        dt,
    ) where {N, T}
    (; coords, index) = chain

    # recast integrator/timestep/grid to the marker precision (see particle advection!);
    # `backend_grid` also makes the grid GPU-safe -- it is indexed directly inside the
    # kernel, so ranges are rebuilt Float32-safe and refined grids are moved to the device
    Tc = eltype(eltype(coords[1]))
    backend = ka_backend(chain)
    method = set_precision(method, Tc)
    dt = convert(Tc, dt)
    grid_vi = backend_grid(backend, grid_vi, Tc)

    # compute some basic stuff
    ni = size(index, 1)

    local_limits = inner_limits(grid_vi)

    # launch parallel advection kernel
    launch!(
        backend, advection_markerchain_kernel!, ni,
        coords, method, V, index, grid_vi, local_limits, dt
    )
    return nothing
end

# DIMENSION AGNOSTIC KERNELS

# Runge-Kutta advection kernel for staggered grids.
@kernel function advection_markerchain_kernel!(
        p,
        method::AbstractAdvectionIntegrator,
        V::NTuple{N, T},
        index,
        grid,
        local_limits,
        dt,
    ) where {N, T}
    i = @index(Global)

    for ipart in cellaxes(index)
        # skip if particle does not exist in this memory location
        doskip(index, ipart, i) && continue
        # extract particle coordinates
        pᵢ = get_particle_coords(p, ipart, i)
        # advect particle
        pᵢ_new = advect_particle_markerchain(method, pᵢ, V, grid, local_limits, dt, i)
        # update particle coordinates
        for k in 1:N
            @inbounds CAI.@index p[k][ipart, i] = pᵢ_new[k]
        end
    end
end

@inline function interp_velocity2particle_markerchain(
        particle_coords::NTuple{N, Any}, grid_vi, local_limits, V::NTuple{N, Any}, icell
    ) where {N}
    return ntuple(Val(N)) do i
        Base.@_inline_meta
        local_lims = local_limits[i]
        v = if check_local_limits(local_lims, particle_coords)
            interp_velocity_grid2particle(particle_coords, grid_vi[i], V[i], icell)
        else
            # Typed sentinel: a bare `Inf` is Float64 and widens the tuple eltype,
            # which forces heap allocation inside the kernel (fatal on Metal).
            convert(eltype(V[i]), Inf)
        end
    end
end

@inline function interp_velocity2particle_markerchain(
        particle_coords::NTuple{N, Any}, grid_vi, V::NTuple{N, Any}, icell
    ) where {N}
    return ntuple(Val(N)) do i
        Base.@_inline_meta
        interp_velocity_grid2particle(particle_coords, grid_vi[i], V[i], icell)
    end
end

# Interpolate velocity from staggered grid to particle
@inline function interp_velocity_grid2particle(
        pᵢ::Union{SVector, NTuple}, xi_vx::NTuple, F::AbstractArray, icell
    )
    # F, coordinates and spacing of the cell corners
    Fi, x_vertex_cell, dxi = corner_field_nodes_MC(F, pᵢ, xi_vx, icell)
    # normalize particle coordinates
    ti = normalize_coordinates(pᵢ, x_vertex_cell, dxi)
    # Interpolate field F onto particle
    Fp = lerp(Fi, ti)
    return Fp
end

# Get field F, and the coordinates and widths of the cell where the particle is located.
# The widths are read off `xi_vx` per direction and per cell, so a staggered grid whose
# components are refined differently is handled by construction.
@inline function corner_field_nodes_MC(F::AbstractArray{T, N}, pᵢ, xi_vx, icell) where {T, N}
    # a marker-chain vertex can sit exactly on the right/top boundary (x == xi_vx[i][end]),
    # where the lookup returns the last vertex; clamp to the last cell so the corner lookup
    # (I and I+1) stays in bounds and interpolates from the edge cell
    I = ntuple(Val(N)) do i
        Base.@_inline_meta
        # the marker's own column is a one- or two-step seed for the horizontal search;
        # the vertical grid has no counterpart, so bisect from its middle
        seed = ifelse(i == 1, icell, midpoint_seed(xi_vx[i]))
        clamp(parent_cell_index(pᵢ[i], xi_vx[i], seed), 1, size(F, i) - 1)
    end

    # coordinates of lower-left corner of the cell
    x_vertex_cell = ntuple(Val(N)) do i
        Base.@_inline_meta
        xi_vx[i][I[i]]
    end
    dxi = ntuple(Val(N)) do i
        Base.@_inline_meta
        cell_width(xi_vx[i], I[i])
    end
    # F at the four centers
    Fi = extract_field_corners(F, I...)

    return Fi, x_vertex_cell, dxi
end


function advect_markerchain!(
    chain::MarkerChain, method::AbstractAdvectionIntegrator, V, grid_vxi, dt
)
    advection!(chain, method, V, grid_vxi, dt)
    move_particles!(chain)
    resample!(chain)

    # interpolate from markers to grid
    compute_topography_vertex!(chain)
    # average h_vertices0 and h_vertices and store in h_vertices
    @. chain.h_vertices = (chain.h_vertices0 + chain.h_vertices) / 2
    # reconstruct chain from vertices
    reconstruct_chain_from_vertices!(chain)
    # update old nodal topography
    copyto!(chain.h_vertices0, chain.h_vertices)

    return nothing
end

# Two-step Runge-Kutta advection scheme for marker chains
function advection!(
    chain::MarkerChain,
    method::AbstractAdvectionIntegrator,
    V,
    grid_vi::NTuple{N,NTuple{N,T}},
    dt,
) where {N,T}
    (; coords, index) = chain

    # compute some basic stuff
    ni = size(index, 1)
    dxi = compute_dx(first(grid_vi))
    # Need to transpose grid_vy and Vy to reuse interpolation kernels

    local_limits = inner_limits(grid_vi)

    # launch parallel advection kernel
    @parallel (@idx ni) advection_markerchain_kernel!(
        coords, method, V, index, grid_vi, local_limits, dxi, dt
    )
    return nothing
end

# DIMENSION AGNOSTIC KERNELS

# ParallelStencil function Runge-Kuttaadvection function for 3D staggered grids
@parallel_indices (i) function advection_markerchain_kernel!(
    p,
    method::AbstractAdvectionIntegrator,
    V::NTuple{N,T},
    index,
    grid,
    local_limits,
    dxi,
    dt,
) where {N,T}
    for ipart in cellaxes(index)
        doskip(index, ipart, i) && continue

        # skip if particle does not exist in this memory location
        doskip(index, ipart, i) && continue
        # extract particle coordinates
        pᵢ = get_particle_coords(p, ipart, i)
        # advect particle
        pᵢ_new = advect_particle_markerchain(method, pᵢ, V, grid, local_limits, dxi, dt)
        # update particle coordinates
        for k in 1:N
            @inbounds @index p[k][ipart, i] = pᵢ_new[k]
        end
    end

    return nothing
end

@inline function interp_velocity2particle_markerchain(
    particle_coords::NTuple{N,Any}, grid_vi, local_limits, dxi, V::NTuple{N,Any}
) where {N}
    return ntuple(Val(N)) do i
        Base.@_inline_meta
        local_lims = local_limits[i]
        v = if check_local_limits(local_lims, particle_coords)
            interp_velocity_grid2particle(particle_coords, grid_vi[i], dxi, V[i])
        else
            Inf
        end
    end
end

# Interpolate velocity from staggered grid to particle
@inline function interp_velocity_grid2particle(
    pᵢ::Union{SVector,NTuple}, xi_vx::NTuple, dxi::NTuple, F::AbstractArray
)
    # F and coordinates at/of the cell corners
    Fi, x_vertex_cell = corner_field_nodes(F, pᵢ, xi_vx, dxi)
    # normalize particle coordinates
    ti = normalize_coordinates(pᵢ, x_vertex_cell, dxi)
    # Interpolate field F onto particle
    Fp = lerp(Fi, ti)
    return Fp
end

# Get field F and nodal indices of the cell corners where the particle is located
@inline function corner_field_nodes(F::AbstractArray{T,N}, pᵢ, xi_vx, dxi) where {T,N}
    I = ntuple(Val(N)) do i
        Base.@_inline_meta
        cell_index(pᵢ[i], xi_vx[i], dxi[1])
    end

    # coordinates of lower-left corner of the cell
    x_vertex_cell = ntuple(Val(N)) do i
        Base.@_inline_meta
        xi_vx[i][I[i]]
    end
    # F at the four centers
    Fi = extract_field_corners(F, I...)

    return Fi, x_vertex_cell
end

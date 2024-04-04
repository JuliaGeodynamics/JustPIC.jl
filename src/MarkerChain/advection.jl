function advect_markerchain!(chain::MarkerChain, V, grid_vx, grid_vy, dt; α::Float64=2 / 3)
    advection_RK!(chain, V, grid_vx, grid_vy, dt, α)
    move_particles!(chain)
    return resample!(chain)
end

"""
    advection_RK!(particles, V, grid_vx, grid_vy, dt, α)

Advect `particles` with the velocity field `V::NTuple{dims, AbstractArray{T,dims}`
on the staggered grid given by `grid_vx` and `grid_vy`using a Runge-Kutta2 scheme
with `α` and time step `dt`.

    xᵢ ← xᵢ + h*( (1-1/(2α))*f(t,xᵢ) + f(t, y+α*h*f(t,xᵢ))) / (2α)
        α = 0.5 ==> midpoint
        α = 1   ==> Heun
        α = 2/3 ==> Ralston
"""

# Two-step Runge-Kutta advection scheme for marker chains
function advection!(
    particles::MarkerChain, 
    method::AbstractAdvectionIntegrator, 
    V, 
    grid_vi::NTuple{N, NTuple{N,T}}, 
    dt
) where {T}
    (; coords, index) = particles

    # compute some basic stuff
    ni = size(index)
    dxi = compute_dx(first(grid_vi))
    # Need to transpose grid_vy and Vy to reuse interpolation kernels

    local_limits = inner_limits(grid_vi)

    # launch parallel advection kernel
    @parallel (@idx ni) advection_markerchain_kernel!(
        coords, methods, V, index, grid_vi, local_limits, dxi, dt
    )
    return nothing
end

# DIMENSION AGNOSTIC KERNELS

# ParallelStencil function Runge-Kuttaadvection function for 3D staggered grids
@parallel_indices (I...) function advection_markerchain_kernel!(
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
        doskip(index, ipart, I...) && continue

        # skip if particle does not exist in this memory location
        doskip(index, ipart, I...) && continue
        # extract particle coordinates
        pᵢ = get_particle_coords(p, ipart, I...)
        # advect particle
        pᵢ_new = advect_particle_markerchain(method, pᵢ, V, grid, local_limits, dxi, dt)
        # update particle coordinates
        for k in 1:N
            @inbounds @cell p[k][ipart, I...] = pᵢ_new[k]
        end
    end

    return nothing
end

@inline function advect_particle_markerchain(
    method::RungeKutta2,
    p0::NTuple{N,T},
    V::NTuple{N,AbstractArray{T,N}},
    grid_vi,
    local_limits,
    dxi,
    dt,
) where {T,N}

    # interpolate velocity to current location
    vp0 = interp_velocity2particle_markerchain(p0, grid_vi, local_limits, dxi, V)

    # first advection stage x = x + v * dt * α
    p1 = first_stage(method, dt, vp0, p0)

    # interpolate velocity to new location
    vp1 = interp_velocity2particle_markerchain(p1, grid_vi, local_limits, dxi, V)

    # final advection step
    p2 = second_stage(method, dt, vp0, vp1, p0)

    return p2
end

@inline function advect_particle_markerchain(
    method::Euler,
    p0::NTuple{N,T},
    V::NTuple{N,AbstractArray{T,N}},
    grid_vi,
    local_limits,
    dxi,
    dt,
) where {T,N}

    # interpolate velocity to current location
    vp0 = interp_velocity2particle_markerchain(p0, grid_vi, local_limits, dxi, V)

    # first advection stage x = x + v * dt
    p1 = first_stage(method, dt, vp0, p0)

    return p1
end

@inline function interp_velocity2particle_markerchain(
    particle_coords::NTuple{N, Any},
    grid_vi, 
    local_limits,
    dxi, 
    V::NTuple{N, Any}, 
) where N
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
    Fp = ndlinear(ti, Fi)
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

function advect_markerchain!(chain::MarkerChain, V, grid_vx, grid_vy, dt; α::Float64 = 2 / 3)
    advection_RK!(chain, V, grid_vx, grid_vy, dt, α)
    move_particles!(chain)
    resample!(chain)
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
function advection_RK!(
    particles::MarkerChain, V, grid_vx::NTuple{2,T}, grid_vy::NTuple{2,T}, dt, α
) where {T}

    (; coords, index) = particles

    # compute some basic stuff   
    ni = size(index)
    dxi = compute_dx(grid_vx)
    # Need to transpose grid_vy and Vy to reuse interpolation kernels
    grid_vi = grid_vx, grid_vy

    local_limits = inner_limits(grid_vi)

    # launch parallel advection kernel
    @parallel (@idx ni) _advection_RK!(coords, V, index, grid_vi, local_limits, dxi, dt, α)
    return nothing
end

function advection_RK!(
    particles::MarkerChain,
    V,
    grid_vx::NTuple{3,T},
    grid_vy::NTuple{3,T},
    grid_vz::NTuple{3,T},
    dt,
    α,
) where {T}
    # unpack 
    (; coords, index) = particles
    # compute some basic stuff
    dxi = compute_dx(grid_vx)
    ni = size(index)

    # Need to transpose grid_vy and Vy to reuse interpolation kernels
    grid_vi = grid_vx, grid_vy, grid_vz
    local_limits = inner_limits(grid_vi)

    # launch parallel advection kernel
    @parallel (@idx ni) _advection_markerchain_RK!(coords, V, index, grid_vi, local_limits, dxi, dt, α)

    return nothing
end

# DIMENSION AGNOSTIC KERNELS

# ParallelStencil fuction Runge-Kuttaadvection function for 3D staggered grids
@parallel_indices (I...) function _advection_markerchain_RK!(
    p, V::NTuple{N,T}, index, grid, local_limits, dxi, dt, α
) where {N,T}
    for ipart in cellaxes(index)
        doskip(index, ipart, I...) && continue

        # cache particle coordinates 
        pᵢ = get_particle_coords(p, ipart, I...)
        p_new = advect_particle_RK(pᵢ, V, grid, local_limits, dxi, dt, α)

        ntuple(Val(N)) do i
            @inbounds @cell p[i][ipart, I...] = p_new[i]
        end
    end

    return nothing
end

function advect_particle_RK(
    p0::NTuple{N,T},
    V::NTuple{N,AbstractArray{T,N}},
    grid_vi,
    local_limits,
    dxi,
    dt,
    α,
) where {T,N}
    ValN = Val(N)
    # interpolate velocity to current location
    vp0 = ntuple(ValN) do i
        Base.@_inline_meta
        local_lims = local_limits[i]

        # if this condition is met, it means that the particle
        # went outside the local rank domain. It will be removed 
        # during shuffling
        v = if check_local_limits(local_lims, p0)
            interp_velocity_grid2particle(p0, grid_vi[i], dxi, V[i])
        else
            zero(T)
        end
    end

    # advect α*dt
    p1 = ntuple(ValN) do i
        Base.@_inline_meta
        muladd(vp0[i], dt * α, p0[i])
    end

    # interpolate velocity to new location
    vp1 = ntuple(ValN) do i
        Base.@_inline_meta
        local_lims = local_limits[i]
        # if this condition is met, it means that the particle
        # went outside the local rank domain. It will be removed 
        # during shuffling
        v = if check_local_limits(local_lims, p1)
            interp_velocity_grid2particle(p1, grid_vi[i], dxi, V[i])
        else
            zero(T)
        end
    end

    # final advection step
    _α = inv(α)
    pf = ntuple(ValN) do i
        Base.@_inline_meta
        if α == 0.5
            @muladd p0[i] + dt * vp1[i]
        else
            @muladd p0[i] + dt * ((1.0 - 0.5 * _α) * vp0[i] + 0.5 * _α * vp1[i])
        end
    end

    return pf
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
@inline function corner_field_nodes(
    F::AbstractArray{T,N}, pᵢ, xi_vx, dxi
) where {T,N}
    # coordinates of lower-left corner of the cell
    x_vertex_cell = ntuple(Val(N)) do i
        Base.@_inline_meta
        I = cell_index(pᵢ, dxi[i])
        xi_vx[i][I]
    end

    # F at the four centers
    Fi = extract_field_corners(F, indices...)

    return Fi, x_vertex_cell
end
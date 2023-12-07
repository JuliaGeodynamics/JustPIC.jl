## 2D SPECIFIC FUNCTIONS 
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
# Main Runge-Kutta advection function for 2D staggered grids

function advection_RK!(
    particles::ClassicParticles, V, grid_vx::NTuple{2,T}, grid_vy::NTuple{2,T}, dt, α
) where {T}
    # unpack 
    (; coords, parent_cell, grid_step) = particles
    np = nparticles(particles)

    # Need to transpose grid_vy and Vy to reuse interpolation kernels
    grid_vi = grid_vx, grid_vy
    local_limits = inner_limits(grid_vi)

    # launch parallel advection kernel
    @parallel (1:np) _advection_classic_RK!(
        coords, V, parent_cell, grid_vi, local_limits, grid_step, dt, α
    )

    return nothing
end

# ParallelStencil fuction Runge-Kutta advection for 2D staggered grids
@parallel_indices (ipart) function _advection_classic_RK!(
    p, V::NTuple{N,T}, parent_cell, grid, local_limits, dxi, dt, α
) where {N, T}
    pᵢ = p[ipart]
    p_new = advect_classic_particle_RK(
        pᵢ, V, parent_cell[ipart], grid, local_limits, dxi, dt, α
    )
    p_SA = SVector{N, eltype(T)}(p_new...)
    p[ipart] = p_SA
    parent_cell[ipart] = get_cell(p_SA, dxi)

    return nothing
end

# DIMENSION AGNOSTIC KERNELS

function advect_classic_particle_RK(
    p0::SVector{N,T},
    V::NTuple{N,AbstractArray{T,N}},
    idx,
    grid_vi,
    local_limits,
    dxi,
    dt,
    α,
) where {T,N}

    _α = inv(α)
    ValN = Val(N)

    # interpolate velocity to current location
    vp0 = ntuple(ValN) do i
        Base.@_inline_meta
        local_lims = local_limits[i]

        # if this condition is met, it means that the particle
        # went outside the local rank domain. It will be removed 
        # during shuffling
        v = if check_local_limits(local_lims, p0) 
            interp_velocity_grid2particle(p0, grid_vi[i], dxi, V[i], idx)
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
    idx2 = get_cell(p1, dxi)
    vp1 = ntuple(ValN) do i
        Base.@_inline_meta
        local_lims = local_limits[i]
        # if this condition is met, it means that the particle
        # went outside the local rank domain. It will be removed 
        # during shuffling
        v = if check_local_limits(local_lims, p1)
            interp_velocity_grid2particle(p1, grid_vi[i], dxi, V[i], idx2)
        else
            zero(T)
        end
    end

    # final advection step
    pf = ntuple(ValN) do i
        Base.@_propagate_inbounds_meta
        Base.@_inline_meta
        if α == 0.5
            @muladd p0[i] + dt * vp1[i]
        else
            @muladd p0[i] + dt * ((1.0 - 0.5 * _α) * vp0[i] + 0.5 * _α * vp1[i])
        end
    end

    return pf
end
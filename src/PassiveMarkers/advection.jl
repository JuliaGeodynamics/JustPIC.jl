function advect_passive_markers!(particles::PassiveMarkers, V, grid_vx, grid_vy, dt; α::Float64=2 / 3)
    advection_RK!(particles, V, grid_vx, grid_vy, dt, α)
    return nothing
end

function advect_passive_markers!(particles::PassiveMarkers, V, grid_vx, grid_vy, grid_vz, dt; α::Float64=2 / 3)
    advection_RK!(particles, V, grid_vx, grid_vy, grid_vz, dt, α)
    return nothing
end

# Two-step Runge-Kutta advection scheme for marker chains
function advection_RK!(
    particles::PassiveMarkers, V, grid_vx::NTuple{2,T}, grid_vy::NTuple{2,T}, dt, α
) where {T}
    (; coords, np) = particles

    # compute some basic stuff
    dxi = compute_dx(grid_vx)
    # Need to transpose grid_vy and Vy to reuse interpolation kernels
    grid_vi = grid_vx, grid_vy

    local_limits = inner_limits(grid_vi)

    # launch parallel advection kernel
    @parallel (1:np) _advection_passive_markers_RK!(coords, V, grid_vi, local_limits, dxi, dt, α)
    return nothing
end

function advection_RK!(
    particles::PassiveMarkers,
    V,
    grid_vx::NTuple{3,T},
    grid_vy::NTuple{3,T},
    grid_vz::NTuple{3,T},
    dt,
    α,
) where {T}
    # unpack
    (; coords, np) = particles
    # compute some basic stuff
    dxi = compute_dx(grid_vx)

    # Need to transpose grid_vy and Vy to reuse interpolation kernels
    grid_vi = grid_vx, grid_vy, grid_vz
    local_limits = inner_limits(grid_vi)

    # launch parallel advection kernel
    @parallel (@idx np) _advection_passive_markers_RK!(
        coords, V, grid_vi, local_limits, dxi, dt, α
    )

    return nothing
end

# DIMENSION AGNOSTIC KERNELS

# ParallelStencil function Runge-Kuttaadvection function for 3D staggered grids
@parallel_indices (ipart) function _advection_passive_markers_RK!(
    p, V::NTuple{N,T}, grid, local_limits, dxi, dt, α
) where {N,T}
    # cache particle coordinates
    pᵢ = get_particle_coords(p, ipart, 1)

    p_new = advect_particle_RK(pᵢ, V, grid, local_limits, dxi, dt, α)

    ntuple(Val(N)) do i
        @inbounds @cell p[i][ipart, 1] = p_new[i]
    end
    return nothing
end

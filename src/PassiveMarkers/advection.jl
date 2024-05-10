# Two-step Runge-Kutta advection scheme for marker chains
function advection!(
    particles::PassiveMarkers, method::AbstractAdvectionIntegrator, V, grid_vxi, dt
)
    (; coords, np) = particles

    # compute some basic stuff
    dxi = compute_dx(first(grid_vxi))
    local_limits = inner_limits(grid_vxi)

    # launch parallel advection kernel
    @parallel (1:np) _advection!(method, coords, V, grid_vxi, local_limits, dxi, dt)
    return nothing
end

# DIMENSION AGNOSTIC KERNELS

# ParallelStencil function Runge-Kutta advection function for 3D staggered grids
@parallel_indices (ipart) function _advection!(
    method::AbstractAdvectionIntegrator, p, V::NTuple{N,T}, grid, local_limits, dxi, dt
) where {N,T}
    # cache particle coordinates 
    pᵢ = get_particle_coords(p, ipart)
    # reuses marker chain methods
    pᵢ_new = advect_particle_markerchain(method, pᵢ, V, grid, local_limits, dxi, dt)

    # p[ipart] = SVector(p_new)
    ntuple(Val(N)) do i
        @inbounds p[i][ipart] = pᵢ_new[i]
    end
    return nothing
end

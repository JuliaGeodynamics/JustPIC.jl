"""
    advection!(particles::PassiveMarkers, method::AbstractAdvectionIntegrator, V, grid_vxi, dt)

Advect passive marker coordinates through the staggered velocity field `V` over a
time step `dt`. The marker coordinates are updated in place.

Unlike the `Particles` method, `grid_vxi` must be supplied explicitly, since
`PassiveMarkers` stores only marker coordinates and no grid metadata.

# Arguments
- `particles`: `PassiveMarkers` container to advect.
- `method`: time integrator such as `Euler()`, `RungeKutta2()`, or `RungeKutta4()`.
- `V`: tuple of staggered velocity component arrays.
- `grid_vxi`: tuple of coordinate tuples matching the staggering of `V`.
- `dt`: timestep.
"""
function advection!(
        particles::PassiveMarkers, method::AbstractAdvectionIntegrator, V, grid_vxi, dt
    )
    (; coords, np) = particles

    # recast integrator/timestep/grid to the marker precision (see particle
    # advection!); `recast_grid` also makes the grid ranges GPU-safe on Float32
    # backends (they are indexed directly inside the kernel)
    Tc = eltype(coords[1])
    method = set_precision(method, Tc)
    dt = convert(Tc, dt)
    grid_vxi = recast_grid(grid_vxi, Tc)

    # compute some basic stuff
    dxi = compute_dx(first(grid_vxi))
    local_limits = inner_limits(grid_vxi)

    # launch parallel advection kernel
    launch!(ka_backend(particles), _advection!, np, method, coords, V, grid_vxi, local_limits, dxi, dt)
    return nothing
end

# DIMENSION AGNOSTIC KERNELS

# Runge-Kutta advection kernel for staggered grids.
@kernel function _advection!(
        method::AbstractAdvectionIntegrator, p, V::NTuple{N}, grid, local_limits, dxi, dt
    ) where {N}
    ipart = @index(Global)

    # cache particle coordinates
    pᵢ = get_particle_coords(p, ipart)
    # reuses marker chain methods
    pᵢ_new = advect_particle_markerchain(method, pᵢ, V, grid, local_limits, dxi, dt)

    # p[ipart] = SVector(p_new)
    ntuple(Val(N)) do i
        @inbounds p[i][ipart] = pᵢ_new[i]
    end
end

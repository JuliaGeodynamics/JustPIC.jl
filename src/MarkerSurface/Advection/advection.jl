"""
    advect_marker_surface!(surf::MarkerSurface, V, xvi, dt;
                           max_slope_angle=0.0,
                           Exx=0.0, Eyy=0.0, bg_ref_x=0.0, bg_ref_y=0.0)

Main driver to advect the free surface:
1. Interpolate velocities from the 3D grid to surface nodes
2. Advect topography using the deformed-grid triangle method
3. Smooth topography spikes (if `max_slope_angle > 0`)

# Arguments
- `surf` : the `MarkerSurface`
- `V`    : tuple `(Vx, Vy, Vz)` of 3D velocity arrays
- `xvi`  : tuple `(xv, yv, zv)` of 1D vertex coordinate arrays
- `dt`   : time step
- `max_slope_angle` : maximum slope angle in **degrees** (default `0.0` = no smoothing)
- `Exx, Eyy` : background strain rates (default `0.0`)
- `bg_ref_x, bg_ref_y` : rotation reference points (default `0.0`)
"""
function advect_marker_surface!(
        surf::MarkerSurface, V::NTuple{3, Any},
        xvi::NTuple{3, Any}, dt;
        max_slope_angle = 45.0,
        Exx = 0.0, Eyy = 0.0
    )
    # Step 1: Interpolate velocities to surface nodes
    interpolate_velocity_to_surface_vertices!(surf, V, xvi)

    # Step 2: Advect topography
    advect_surface_topo!(surf, dt; Exx = Exx, Eyy = Eyy)

    # Step 3: Smooth topography spikes
    smooth_surface_max_angle!(surf, max_slope_angle)

    return nothing
end

@parallel_indices (i, j) function _semilagrangian_update_kernel!(
        topo, vz, dt
    )
    topo[i, j] += vz[i, j] * dt
    return nothing
end

"""
    semilagrangian_advect_surface!(surf::MarkerSurface, V, xvi, dt;
                                   max_slope_angle=45.0)

Semi-Lagrangian surface advection: each vertex is updated by `z += vz*dt`,
followed by slope smoothing and mass conservation.

# Arguments
- `surf` : the `MarkerSurface`
- `V`    : tuple `(Vx, Vy, Vz)` of 3D velocity arrays
- `xvi`  : tuple `(xv, yv, zv)` of 1D vertex coordinate arrays
- `dt`   : time step
- `max_slope_angle` : maximum slope angle in **degrees** (default `45.0`)
"""
function semilagrangian_advect_surface!(
        surf::MarkerSurface, V::NTuple{3, Any},
        xvi::NTuple{3, Any}, dt;
        max_slope_angle = 45.0,
    )
    # Save current state
    copyto!(surf.topo0, surf.topo)
    avg_old = compute_avg_topo(surf)

    # Interpolate velocities to surface
    interpolate_velocity_to_surface_vertices!(surf, V, xvi)

    # Update each vertex: z += vz*dt
    nx1, ny1 = size(surf.topo)

    @parallel (1:nx1, 1:ny1) _semilagrangian_update_kernel!(
        surf.topo, surf.vz, dt
    )

    # Smooth
    smooth_surface_max_angle!(surf, max_slope_angle)

    # Mass conservation: preserve mean height
    avg_new = compute_avg_topo(surf)
    surf.topo .+= (avg_old - avg_new)

    return nothing
end

"""
    advect_marker_surface!(surf::MarkerSurface, V, grid_vxi, dt;
                           max_slope_angle=45)

Main driver to advect the free surface:
1. Interpolate velocities from the 3D grid to surface nodes
2. Advect topography using the deformed-grid triangle method
3. Smooth topography spikes (if `max_slope_angle > 0`)

# Arguments
- `surf` : the `MarkerSurface`
- `V`    : tuple `(Vx, Vy, Vz)` of 3D velocity arrays
- `grid_vxi` : tuple of component grids `(grid_vx, grid_vy, grid_vz)`
- `dt`   : time step
- `max_slope_angle` : maximum slope angle in **degrees** (default `45`; `≤ 0` disables smoothing)
"""
function advect_marker_surface!(
        surf::MarkerSurface, V::NTuple{3, Any},
        grid_vxi::NTuple{3, Any}, dt;
        max_slope_angle = 45,
    )
    T = eltype(surf.topo)
    dt = convert(T, dt)
    max_slope_angle = convert(T, max_slope_angle)
    # Step 1: Interpolate velocities to surface nodes
    interpolate_velocity_to_surface_vertices!(surf, V, grid_vxi)

    # Step 2: Advect topography
    advect_surface_topo!(surf, dt)

    # Step 3: Smooth topography spikes
    smooth_surface_max_angle!(surf, max_slope_angle)

    return nothing
end

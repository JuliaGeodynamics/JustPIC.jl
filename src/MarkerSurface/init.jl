"""
    init_marker_surface(::Type{backend}, xv, yv, initial_elevation;
                        air_phase=0, periodic_1=false, periodic_2=false)

Create a `MarkerSurface` that tracks a 3D free surface on the grid defined by
vertex coordinates `xv` and `yv`.

The topography is stored at the grid vertices (corner nodes), matching LaMEM's
`FreeSurf` approach where the surface DMDA has the same (x,y)-resolution as the
staggered-grid corner nodes.

# Arguments
- `backend` : KernelAbstractions backend type such as `CPU`, `CUDA`, `AMDGPU`, `Metal`
- `xv`      : 1D array/range of x-coordinates of grid vertices (length `nx+1`)
- `yv`      : 1D array/range of y-coordinates of grid vertices (length `ny+1`)
- `initial_elevation` : scalar or 2D array `(nx+1)×(ny+1)` of initial z-elevations
- `air_phase`   : phase ID for the sticky-air layer (default `0`)
- `periodic_1`, `periodic_2` : periodic boundary conditions in x and y (default `false`)

# Returns
A `MarkerSurface` instance with topography initialised to `initial_elevation`.
"""
function init_marker_surface(
        ::Type{backend}, xv, yv, initial_elevation;
        air_phase::Int = 0,
        periodic_1::Bool = false,
        periodic_2::Bool = false,
    ) where {backend}

    nx1 = length(xv)
    ny1 = length(yv)

    # Allocate topography arrays at vertices
    topo = TA(backend)(fill(zero(Float64), nx1, ny1))
    topo0 = TA(backend)(fill(zero(Float64), nx1, ny1))
    vx = TA(backend)(zeros(Float64, nx1, ny1))
    vy = TA(backend)(zeros(Float64, nx1, ny1))
    vz = TA(backend)(zeros(Float64, nx1, ny1))

    # Set initial topography
    if initial_elevation isa Number
        fill!(topo, Float64(initial_elevation))
        fill!(topo0, Float64(initial_elevation))
    else
        copyto!(topo, initial_elevation)
        copyto!(topo0, initial_elevation)
    end

    xv_arr = TA(backend)(collect(Float64, xv))
    yv_arr = TA(backend)(collect(Float64, yv))

    return MarkerSurface(
        backend,
        topo, topo0,
        vx, vy, vz,
        xv_arr, yv_arr,
        air_phase, periodic_1, periodic_2,
    )
end

"""
    compute_avg_topo(surf::MarkerSurface)

Compute and return the average topography over all surface vertices. Duplicated
periodic seam nodes are counted once. Under MPI the average is **rank-local**
(the rank-local array, including halo vertices); JustPIC does no global
reductions.
Note: forces a device→host scalar transfer on GPU; call only outside hot loops.
"""
function compute_avg_topo(surf::MarkerSurface)
    nx1, ny1 = size(surf.topo)
    i_end = surf.periodic_1 ? nx1 - 1 : nx1
    j_end = surf.periodic_2 ? ny1 - 1 : ny1
    t = @view surf.topo[1:i_end, 1:j_end]
    return sum(t) / length(t)
end

"""
    set_topo_from_array!(surf::MarkerSurface, z::AbstractMatrix)

Set the surface topography from a 2D array `z` of size `(nx+1, ny+1)`.
Also copies the values into `topo0`.
"""
function set_topo_from_array!(surf::MarkerSurface, z::AbstractMatrix)
    @assert size(z) == size(surf.topo) "Topography array size must match surface grid size"
    copyto!(surf.topo, z)
    copyto!(surf.topo0, z)
    return nothing
end

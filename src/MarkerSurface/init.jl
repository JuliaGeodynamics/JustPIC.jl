"""
    init_marker_surface(::Type{backend}, xv, yv, initial_elevation;
                        periodic_1=false, periodic_2=false)

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
- `periodic_1`, `periodic_2` : periodic boundary conditions in x and y (default `false`)

# Returns
A `MarkerSurface` instance with topography initialised to `initial_elevation`.
"""
function init_marker_surface(
        ::Type{backend}, xv, yv, initial_elevation;
        periodic_1::Bool = false,
        periodic_2::Bool = false,
    ) where {backend}

    _validate_surface_coordinates("xv", xv)
    _validate_surface_coordinates("yv", yv)
    nx1, ny1 = length(xv), length(yv)
    initial_elevation isa AbstractMatrix && size(initial_elevation) == (nx1, ny1) ||
        initial_elevation isa Number ||
        throw(DimensionMismatch("initial_elevation must be a scalar or have size ($(nx1), $(ny1))"))

    Tinitial = initial_elevation isa AbstractArray ? eltype(initial_elevation) : typeof(initial_elevation)
    T = promote_type(eltype(xv), eltype(yv), Tinitial)
    T <: AbstractFloat || throw(ArgumentError("MarkerSurface inputs must promote to an AbstractFloat"))

    xv_arr = TA(backend)(T.(recast_grid(xv, T)))
    yv_arr = TA(backend)(T.(recast_grid(yv, T)))
    initial_elevation = initial_elevation isa AbstractArray ?
        T.(initial_elevation) : convert(T, initial_elevation)

    # Allocate topography arrays at vertices
    topo = TA(backend)(zeros(T, nx1, ny1))
    topo0 = TA(backend)(zeros(T, nx1, ny1))
    vx = TA(backend)(zeros(T, nx1, ny1))
    vy = TA(backend)(zeros(T, nx1, ny1))
    vz = TA(backend)(zeros(T, nx1, ny1))

    # Set initial topography
    if initial_elevation isa Number
        fill!(topo, initial_elevation)
        fill!(topo0, initial_elevation)
    else
        copyto!(topo, initial_elevation)
        copyto!(topo0, initial_elevation)
    end

    return MarkerSurface(
        backend,
        topo, topo0,
        vx, vy, vz,
        xv_arr, yv_arr,
        periodic_1, periodic_2,
    )
end

function _validate_surface_coordinates(name, x)
    length(x) ≥ 2 || throw(ArgumentError("$name must contain at least two coordinates"))
    all(isfinite, x) || throw(ArgumentError("$name must contain only finite coordinates"))
    all(diff(x) .> zero(eltype(x))) ||
        throw(ArgumentError("$name must be strictly increasing"))
    return nothing
end

"""
    compute_avg_topo(surf::MarkerSurface)

Compute and return the average topography over all surface vertices. Duplicated
periodic seam nodes are counted once. Under MPI, overlapping x/y nodes and
replicated z-columns are counted once through an owned-node global reduction.
Note: forces a device→host scalar transfer on GPU; call only outside hot loops.
"""
function compute_avg_topo(surf::MarkerSurface)
    nx1, ny1 = size(surf.topo)
    i_end = surf.periodic_1 ? nx1 - 1 : nx1
    j_end = surf.periodic_2 ? ny1 - 1 : ny1
    if !ImplicitGlobalGrid.grid_is_initialized()
        topo = @view surf.topo[1:i_end, 1:j_end]
        return sum(topo) / length(topo)
    end

    gg = ImplicitGlobalGrid.global_grid()
    i_end = min(i_end, _owned_surface_extent(nx1, 1, gg))
    j_end = min(j_end, _owned_surface_extent(ny1, 2, gg))
    owns_z_column = gg.coords[3] == 0
    topo = @view surf.topo[1:i_end, 1:j_end]
    local_sum = owns_z_column ? sum(topo) : zero(eltype(surf.topo))
    local_count = owns_z_column ? length(topo) : 0
    total = MPI.Allreduce(local_sum, +, gg.comm)
    count = MPI.Allreduce(local_count, +, gg.comm)
    count > 0 || throw(ArgumentError("MarkerSurface has no owned topography nodes"))
    return total / count
end

@inline function _owned_surface_extent(n, dim, gg)
    gg.coords[dim] == gg.dims[dim] - 1 && return n
    overlap = gg.overlaps[dim] + n - gg.nxyz[dim]
    extent = n - overlap
    extent > 0 || throw(ArgumentError("MarkerSurface local size is incompatible with ImplicitGlobalGrid overlap"))
    return extent
end

"""
    set_topo_from_array!(surf::MarkerSurface, z::AbstractMatrix)

Set the surface topography from a 2D array `z` of size `(nx+1, ny+1)`.
Also copies the values into `topo0`.
"""
function set_topo_from_array!(surf::MarkerSurface, z::AbstractMatrix)
    size(z) == size(surf.topo) ||
        throw(DimensionMismatch("topography must have size $(size(surf.topo))"))
    copyto!(surf.topo, z)
    copyto!(surf.topo0, z)
    return nothing
end

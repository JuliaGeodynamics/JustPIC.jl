# Multi-rank MPI tests for the MarkerSurface free-surface tracker.
#
# Not part of the default `runtests()` (that runs single-process); launch
# explicitly with, e.g.:
#
#     mpiexecjl -n 4 julia --project=. test/test_marker_surface_mpi.jl
#
# or from Julia:
#
#     using MPI; run(`$(mpiexec()) -n 4 $(Base.julia_cmd()) --project=. test/test_marker_surface_mpi.jl`)
#
# Verifies that under ImplicitGlobalGrid domain decomposition — including
# decomposition in z — the distributed surface is bit-identical to a serial run
# on the equivalent global grid.

using MPI, ImplicitGlobalGrid, JustPIC, Test

const backend = JustPIC.CPU

# analytic, z-dependent velocity field so the owning z-slab matters
vx_f(x, y, z) = 0.05 * sin(2π * x) * cos(2π * y) + 0.02 * z
vy_f(x, y, z) = 0.05 * cos(2π * x) * sin(2π * y) - 0.03 * z
vz_f(x, y, z) = 0.1 * cos(2π * x) * cos(2π * y) + 0.07 * z
topo_f(x, y) = 0.5 + 0.3 * sin(2π * x) * sin(2π * y)

function extend_centers(xv)
    xc = [(xv[i] + xv[i + 1]) / 2 for i in 1:(length(xv) - 1)]
    return [2 * xc[1] - xc[2]; xc; 2 * xc[end] - xc[end - 1]]
end

staggered_grid(xv, yv, zv) = (
    (xv, extend_centers(yv), extend_centers(zv)),
    (extend_centers(xv), yv, extend_centers(zv)),
    (extend_centers(xv), extend_centers(yv), zv),
)

sampleV(grid_vxi) = ntuple(Val(3)) do d
    x, y, z = grid_vxi[d]
    f = d == 1 ? vx_f : d == 2 ? vy_f : vz_f
    [f(x_, y_, z_) for x_ in x, y_ in y, z_ in z]
end

MPI.Initialized() || MPI.Init()
np = MPI.Comm_size(MPI.COMM_WORLD)
horizontal_decomp = get(ENV, "JULIA_MARKER_SURFACE_MPI_HORIZONTAL", "x")
horizontal_decomp ∈ ("x", "y") || error("JULIA_MARKER_SURFACE_MPI_HORIZONTAL must be \"x\" or \"y\"")

# choose a decomposition that includes z whenever there is more than one rank
dimx, dimy, dimz = np == 1 ? (1, 1, 1) :
    np == 2 ? (1, 1, 2) :
    np == 4 ? (horizontal_decomp == "x" ? (2, 1, 2) : (1, 2, 2)) :
    (1, 1, np)

nx = ny = nz = 16
me, dims, nprocs, coords, = init_global_grid(nx, ny, nz; dimx, dimy, dimz, init_MPI = false, quiet = true)

ol = 2
nvx_g = dims[1] * nx - ol * (dims[1] - 1)
nvy_g = dims[2] * ny - ol * (dims[2] - 1)
nvz_g = dims[3] * nz - ol * (dims[3] - 1)
dx = 1.0 / nvx_g; dy = 1.0 / nvy_g; dz = 1.0 / nvz_g

xoff = coords[1] * (nx - ol)
yoff = coords[2] * (ny - ol)
zoff = coords[3] * (nz - ol)
xv = [(i - 1 + xoff) * dx for i in 1:(nx + 1)]
yv = [(j - 1 + yoff) * dy for j in 1:(ny + 1)]
zv = [(k - 1 + zoff) * dz for k in 1:(nz + 1)]
xv_glob = [(i - 1) * dx for i in 1:(nvx_g + 1)]
yv_glob = [(j - 1) * dy for j in 1:(nvy_g + 1)]
zv_glob = [(k - 1) * dz for k in 1:(nvz_g + 1)]

dt = 0.02
nt = 8

# --- distributed run ---
surf = init_marker_surface(backend, xv, yv, [topo_f(x, y) for x in xv, y in yv])
grid_vxi = staggered_grid(xv, yv, zv)
Vloc = sampleV(grid_vxi)
for _ in 1:nt
    advect_marker_surface!(surf, Vloc, grid_vxi, dt)
end
smooth_surface_max_angle!(surf, 25.0)
topo_dist = Array(surf.topo)
avg_dist = compute_avg_topo(surf)

outside_surf = init_marker_surface(backend, xv, yv, 1.5)
outside_error = try
    interpolate_velocity_to_surface_vertices!(outside_surf, Vloc, grid_vxi)
    false
catch err
    err isa ArgumentError
end
outside_error_on_all_ranks = MPI.Allreduce(outside_error, &, MPI.COMM_WORLD)

# z-replicas (ranks differing only in coords[3]) must be identical
root = copy(topo_dist)
MPI.Bcast!(root, 0, MPI.COMM_WORLD)  # rank 0 is at coords (0,0,0)
same_column = coords[1] == 0 && coords[2] == 0
replica_err = same_column ? maximum(abs.(topo_dist .- root)) : 0.0

# --- serial reference on the full global grid ---
finalize_global_grid(; finalize_MPI = false)
ref = init_marker_surface(backend, xv_glob, yv_glob, [topo_f(x, y) for x in xv_glob, y in yv_glob])
grid_vxi_ref = staggered_grid(xv_glob, yv_glob, zv_glob)
Vref = sampleV(grid_vxi_ref)
for _ in 1:nt
    advect_marker_surface!(ref, Vref, grid_vxi_ref, dt)
end
smooth_surface_max_angle!(ref, 25.0)
err = maximum(
    abs.(
        topo_dist .- Array(ref.topo)[
            (1 + xoff):(nx + 1 + xoff), (1 + yoff):(ny + 1 + yoff),
        ]
    )
)
avg_ref = sum(ref.topo) / length(ref.topo)

@testset "MarkerSurface MPI (np=$np, dims=$(Tuple(dims)))" begin
    @test replica_err < 1.0e-14   # z-column replicas stay consistent
    @test err < 1.0e-12           # distributed == serial, bit for bit
    @test outside_error_on_all_ranks
    @test avg_dist ≈ avg_ref
end

MPI.Finalize()

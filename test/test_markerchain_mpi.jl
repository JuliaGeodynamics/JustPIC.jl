# Multi-rank MPI tests for the MarkerChain free-surface tracker.
#
# Not part of the default `runtests()` (that runs single-process); launch
# explicitly with, e.g.:
#
#     mpiexecjl -n 4 julia --project=. test/test_markerchain_mpi.jl
#
# `JULIA_MARKERCHAIN_MPI_DIMS` selects the decomposition of 4 ranks: "x" (4x1),
# "y" (1x4) or "xy" (2x2, the default). Any other rank count splits along x.

using MPI, ImplicitGlobalGrid, JustPIC, Test

const backend = JustPIC.CPU

MPI.Initialized() || MPI.Init()
# an error on one rank would leave the others blocked in a collective; end the whole job
atexit(code -> iszero(code) || MPI.Abort(MPI.COMM_WORLD, code))
np = MPI.Comm_size(MPI.COMM_WORLD)
decomposition = get(ENV, "JULIA_MARKERCHAIN_MPI_DIMS", "xy")
decomposition ∈ ("x", "y", "xy") || error("JULIA_MARKERCHAIN_MPI_DIMS must be \"x\", \"y\" or \"xy\"")
dimx, dimy = np == 4 ? (decomposition == "x" ? (4, 1) : decomposition == "y" ? (1, 4) : (2, 2)) : (np, 1)

nx, ny = 18, 14 # local cells
me, dims, nprocs, coords, comm = init_global_grid(nx, ny, 1; dimx, dimy, init_MPI = false, quiet = true)
xoff = coords[1] * (nx - 2)
yoff = coords[2] * (ny - 2)
xrow = MPI.Cart_sub(comm, (true, false, false))

# global vertex coordinates, refined towards the right and the top
xv_g = cumsum([0.0; range(1.0, 2.0; length = nx_g())]) ./ (1.5 * nx_g())
yv_g = cumsum([0.0; range(2.0, 1.0; length = ny_g())]) ./ (1.5 * ny_g())
xv = xv_g[xoff .+ (1:(nx + 1))]
yv = yv_g[yoff .+ (1:(ny + 1))]

@testset "MarkerChain MPI helpers (np=$np, dims=$(Tuple(dims)))" begin
    # x-halo of a vertex vector: the outer entries come from the neighboring ranks
    h = Float64.(xoff .+ (1:(nx + 1)))
    first_rank, last_rank = coords[1] == 0, coords[1] == dims[1] - 1
    first_rank || (h[1] = -1)
    last_rank || (h[end] = -1)
    JustPIC._update_chain_halo!(h)
    @test h == Float64.(xoff .+ (1:(nx + 1)))

    # every global cell along x is owned by exactly one rank of an x-row
    @test MPI.Allreduce(length(JustPIC._owned_cells(nx)), +, xrow) == nx_g()

    # the mean height over the owned cells equals the serial mean of the global chain
    h_g = 0.3 .+ 0.1 .* sin.(4π .* xv_g)
    h_loc = h_g[xoff .+ (1:(nx + 1))]
    @test JustPIC._global_mean_height(h_loc, xv) ≈ JustPIC.mean_height(h_g, xv_g) rtol = 1.0e-13
end

# --- semi-Lagrangian advection: distributed chain vs serial chain on the global grid ---
vx_f(x, y) = 0.2 * sin(π * x) + 0.1 * y
vy_f(x, y) = 0.1 * cos(π * x) + 0.05 * y
h0_f(x) = 0.5 + 0.2 * sin(2π * x)

centers(v) = (v[1:(end - 1)] .+ v[2:end]) ./ 2
extend_centers(v) = (c = centers(v); [2 * c[1] - c[2]; c; 2 * c[end] - c[end - 1]])
# velocity grids of the global domain; the local ones are slices, so their ghost nodes
# are the true centers of the neighboring cells
grid_vxi_g = ((xv_g, extend_centers(yv_g)), (extend_centers(xv_g), yv_g))
grid_vxi = (
    (xv, grid_vxi_g[1][2][yoff .+ (1:(ny + 2))]),
    (grid_vxi_g[2][1][xoff .+ (1:(nx + 2))], yv),
)
sampleV(grid_vxi, fx = vx_f, fy = vy_f) = (
    [fx(x, y) for x in grid_vxi[1][1], y in grid_vxi[1][2]],
    [fy(x, y) for x in grid_vxi[2][1], y in grid_vxi[2][2]],
)
dt = 0.5 * minimum(diff(xv_g)) / 0.3
nt = 10
cases = ((;), (; conserve_mean = false), (; max_slope_angle = nothing))

function run_chain(xv, yv, grid_vxi, V; kwargs...)
    chain = init_markerchain(backend, 6, 4, 8, xv, h0_f.(xv))
    for _ in 1:nt
        semilagrangian_advection_markerchain!(chain, RungeKutta2(), V, grid_vxi, (xv, yv), dt; kwargs...)
    end
    return chain
end

V = sampleV(grid_vxi)
chains = map(kw -> run_chain(xv, yv, grid_vxi, V; kw...), cases)
h_dist = map(c -> Array(c.h_vertices), chains)
# the markers are those rebuilt from the vertex heights
markers_consistent = all(chains) do chain
    rebuilt = init_markerchain(backend, 6, 4, 8, xv, 0.0)
    fill_chain_from_vertices!(rebuilt, chain.h_vertices)
    all(d -> isequal(chain.coords[d].data, rebuilt.coords[d].data), 1:2)
end

# a chain above the domain has no velocity on any rank
outside = init_markerchain(backend, 6, 4, 8, xv, 2 * yv_g[end])
outside_msg = try
    semilagrangian_advection_markerchain!(outside, RungeKutta2(), V, grid_vxi, (xv, yv), dt)
    ""
catch err
    sprint(showerror, err)
end
outside_on_all_ranks = MPI.Allreduce(occursin("outside every local vertical velocity slab", outside_msg), &, comm)

# a uniform flow moving two cells per step cannot be remapped from the local chain
far = init_markerchain(backend, 6, 4, 8, xv, h0_f.(xv))
Vfar = sampleV(grid_vxi, (x, y) -> 1.0, (x, y) -> 0.0)
far_msg = try
    semilagrangian_advection_markerchain!(far, RungeKutta2(), Vfar, grid_vxi, (xv, yv), 2 * maximum(diff(xv_g)))
    ""
catch err
    sprint(showerror, err)
end
far_on_all_ranks = MPI.Allreduce(occursin("more than one cell along x", far_msg), &, comm)

# the same along y: an owned vertex leaves its rank's local velocity grid
far_y = init_markerchain(backend, 6, 4, 8, xv, h0_f.(xv))
Vfar_y = sampleV(grid_vxi, (x, y) -> 0.0, (x, y) -> 1.0)
far_y_msg = try
    semilagrangian_advection_markerchain!(far_y, RungeKutta2(), Vfar_y, grid_vxi, (xv, yv), 2 * maximum(diff(yv_g)))
    ""
catch err
    sprint(showerror, err)
end
far_y_on_all_ranks = MPI.Allreduce(occursin("local grid along y", far_y_msg), &, comm)

# rock fractions of a local chain, halo entries included
function rock_fractions(xv, yv, nx, ny)
    chain = init_markerchain(backend, 6, 4, 8, xv, h0_f.(xv))
    ratios = (;
        center = zeros(nx, ny), vertex = zeros(nx + 1, ny + 1),
        Vx = zeros(nx + 1, ny), Vy = zeros(nx, ny + 1),
    )
    compute_rock_fraction!(ratios, chain, (xv, yv), (diff(xv), diff(yv)))
    return ratios
end
ratios_dist = rock_fractions(xv, yv, nx, ny)

# the Lagrangian chain path has no particle exchange between ranks
lagrangian = init_markerchain(backend, 6, 4, 8, xv, h0_f.(xv))
lagrangian_msgs = map(
    (
        () -> advect_markerchain!(lagrangian, RungeKutta2(), V, grid_vxi, dt),
        () -> move_particles!(lagrangian),
    )
) do f
    try
        f()
        ""
    catch err
        sprint(showerror, err)
    end
end
lagrangian_rejected = all(m -> occursin("use semilagrangian_advection_markerchain!", m), lagrangian_msgs)

finalize_global_grid(; finalize_MPI = false)

# serial reference on the full global grid
V_g = sampleV(grid_vxi_g)
h_serial = map(kw -> Array(run_chain(xv_g, yv_g, grid_vxi_g, V_g; kw...).h_vertices), cases)
errs = map((hd, hs) -> maximum(abs.(hd .- hs[xoff .+ (1:(nx + 1))])), h_dist, h_serial)
# a periodic x direction is rejected
init_global_grid(nx, ny, 1; dimx, dimy, periodx = 1, init_MPI = false, quiet = true)
periodic_chain = init_markerchain(backend, 6, 4, 8, xv, h0_f.(xv))
periodic_msg = try
    semilagrangian_advection_markerchain!(periodic_chain, RungeKutta2(), V, grid_vxi, (xv, yv), dt)
    ""
catch err
    sprint(showerror, err)
end
finalize_global_grid(; finalize_MPI = false)

# the ownership rules assume the default overlap
init_global_grid(nx, ny, 1; dimx, dimy, overlaps = (4, 4, 2), init_MPI = false, quiet = true)
overlap_chain = init_markerchain(backend, 6, 4, 8, xv, h0_f.(xv))
overlap_msg = try
    semilagrangian_advection_markerchain!(overlap_chain, RungeKutta2(), V, grid_vxi, (xv, yv), dt)
    ""
catch err
    sprint(showerror, err)
end
finalize_global_grid(; finalize_MPI = false)

ratios_serial = rock_fractions(xv_g, yv_g, length(xv_g) - 1, length(yv_g) - 1)
ratio_errs = map(keys(ratios_dist)) do k
    local_ratio = ratios_dist[k]
    global_ratio = ratios_serial[k][xoff .+ axes(local_ratio, 1), yoff .+ axes(local_ratio, 2)]
    maximum(abs.(local_ratio .- global_ratio))
end

@testset "MarkerChain MPI semi-Lagrangian (np=$np, dims=$(Tuple(dims)))" begin
    for (case, err) in zip(cases, errs)
        @test err < 1.0e-12
    end
    np == 1 && @test all(iszero, errs)
    @test markers_consistent
    @test outside_on_all_ranks
    @test dims[1] == 1 || far_on_all_ranks
    @test dims[2] == 1 || far_y_on_all_ranks
end

@testset "MarkerChain MPI unsupported paths (np=$np, dims=$(Tuple(dims)))" begin
    @test lagrangian_rejected == (np > 1)
    @test occursin("periodic x", periodic_msg)
    @test np == 1 || occursin("overlaps of 2", overlap_msg)
end

@testset "MarkerChain MPI rock fractions (np=$np, dims=$(Tuple(dims)))" begin
    for (k, err) in zip(keys(ratios_dist), ratio_errs)
        @test err < 1.0e-12
    end
end

MPI.Finalize()

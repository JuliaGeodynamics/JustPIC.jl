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
far_on_all_ranks = MPI.Allreduce(occursin("more than one cell", far_msg), &, comm)

finalize_global_grid(; finalize_MPI = false)

# serial reference on the full global grid
V_g = sampleV(grid_vxi_g)
h_serial = map(kw -> Array(run_chain(xv_g, yv_g, grid_vxi_g, V_g; kw...).h_vertices), cases)
errs = map((hd, hs) -> maximum(abs.(hd .- hs[xoff .+ (1:(nx + 1))])), h_dist, h_serial)

@testset "MarkerChain MPI semi-Lagrangian (np=$np, dims=$(Tuple(dims)))" begin
    for (case, err) in zip(cases, errs)
        @test err < 1.0e-12
    end
    np == 1 && @test all(iszero, errs)
    @test markers_consistent
    @test outside_on_all_ranks
    @test dims[1] == 1 || far_on_all_ranks
end

MPI.Finalize()

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

finalize_global_grid(; finalize_MPI = false)
MPI.Finalize()

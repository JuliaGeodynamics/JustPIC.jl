# Multi-rank MPI test for `update_cell_halo!` on particle CellArrays.
#
# Not part of the default `runtests()` (that runs single-process); launch
# explicitly with, e.g.:
#
#     mpiexecjl -n 4 julia --project=. test/test_cell_halo_mpi.jl
#
# 4 ranks split 2x2; any other rank count splits along x.

using MPI, ImplicitGlobalGrid, JustPIC, Test
import JustPIC: CAI

const backend = JustPIC.CPU

MPI.Initialized() || MPI.Init()
# an error on one rank would leave the others blocked in a collective; end the whole job
atexit(code -> iszero(code) || MPI.Abort(MPI.COMM_WORLD, code))
np = MPI.Comm_size(MPI.COMM_WORLD)
dimx, dimy = np == 4 ? (2, 2) : (np, 1)

nx, ny = 8, 6 # local cells; IGG default overlaps = 2
me, dims, nprocs, coords, comm = init_global_grid(nx, ny, 1; dimx, dimy, init_MPI = false, quiet = true)
offset = (coords[1] * (nx - 2), coords[2] * (ny - 2))

xv = offset[1] .+ collect(0.0:nx)
yv = offset[2] .+ collect(0.0:ny)
xc, yc = xv[1:(end - 1)] .+ 0.5, yv[1:(end - 1)] .+ 0.5
grid_vx = (xv, [yc[1] - 1; yc; yc[end] + 1])
grid_vy = ([xc[1] - 1; xc; xc[end] + 1], yv)
particles = init_particles(backend, 4, 8, 2, grid_vx, grid_vy; periodic = (false, false))
pRank, = init_cell_arrays(particles, Val(1))
pRank.data .= me

@testset "update_cell_halo! halo widths" begin
    @test JustPIC.cell_halowidths(size(particles.index)) == (2, 2, 1)
    @test JustPIC.cell_halowidths((nx, ny)) == (1, 1, 1)
    @test JustPIC.cell_halowidths((nx + 1, ny + 1)) == (1, 1, 1)
end

update_cell_halo!(particles.coords..., pRank)
update_cell_halo!(particles.index)

# particles of every local cell, keyed by global cell; particle cell `i` (ghost at 1 and
# n + 2) is global cell `offset + i - 1`
(; index) = particles
px, py = particles.coords
local_cells = Dict{NTuple{2, Int}, Vector{NTuple{3, Float64}}}()
for j in axes(index, 2), i in axes(index, 1)
    cell = [
        (CAI.@index(px[ip, i, j]), CAI.@index(py[ip, i, j]), CAI.@index(pRank[ip, i, j]))
            for ip in JustPIC.cellaxes(index) if CAI.@index(index[ip, i, j])
    ]
    local_cells[(offset[1] + i - 1, offset[2] + j - 1)] = sort(cell)
end

gathered = MPI.gather(local_cells, comm; root = 0)
mismatches = if me == 0
    seen = Dict{NTuple{2, Int}, Vector{NTuple{3, Float64}}}()
    sum(gathered) do cells
        count(cells) do (g, cell)
            haskey(seen, g) || (seen[g] = cell; return false)
            return seen[g] != cell
        end
    end
else
    0
end
mismatches = MPI.bcast(mismatches, 0, comm)
nshared = MPI.Allreduce(length(local_cells), +, comm) - (dims[1] * (nx - 2) + 4) * (dims[2] * (ny - 2) + 4)

@testset "update_cell_halo! synchronizes every shared cell" begin
    @test np == 1 || nshared > 0
    @test mismatches == 0
end

finalize_global_grid(; finalize_MPI = false)

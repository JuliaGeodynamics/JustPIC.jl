using JustPIC
import JustPIC._2D as JP2
import JustPIC._3D as JP3
using Test

function init_2D_particles(backend)
    nxcell, max_xcell, min_xcell = 4, 4, 4
    n = 2
    nx = ny = n-1
    Lx = Ly = 1.0
    # nodal vertices
    xvi = xv, yv = LinRange(0, Lx, n), LinRange(0, Ly, n)
    dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]

    particles = JP2.init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi...,
    )
    particles, xvi
end


@testset "interpolation utils" begin
    # 2D
    xi2 = x, y = LinRange(0, 1, 10), LinRange(0, 1, 10)
    pxi2 = px, py = 0.5, 0.5
    @test JP2.isinside(pxi2, xi2)

    pxi = rand(10), rand(10)
    @test JP2.particle2tuple(pxi, 10) == (pxi[1][10], pxi[2][10])

    A = rand(2, 2)
    @test JP2.field_corners(A, (1, 1)) == tuple(A[:]...)

    # 3D
    xi3 = (xi2..., LinRange(0, 1, 10))
    pxi3 = (pxi2..., 0.5)
    @test JP2.isinside(pxi3, xi3)

    pxi = rand(10), rand(10), rand(10)
    @test JP2.particle2tuple(pxi, 10) == (pxi[1][10], pxi[2][10], pxi[3][10])

    A = rand(2, 2, 2)
    @test JP2.field_corners(A, (1, 1, 1)) == tuple(A[:]...)
end


# particles, xvi = init_2D_particles(JP2.CPUBackend)
# A = B          = [
#     0 0
#     1 1
# ]
# particle_args = pA, pB = JP2.init_cell_arrays(particles, Val(2));

# JP2.grid2particle!(pA, xvi, A, particles)
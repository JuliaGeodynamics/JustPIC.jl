using Test
using JustPIC, JustPIC._2D
using LinearAlgebra
import JustPIC._2D: lerp

const backend = JustPIC.CPUBackend

@testset "Interpolation kernels" begin
    @testset "lerp" begin
        t1D = (0.5,)
        v1D = 1.0e0, 2.0e0

        @test lerp(v1D, t1D) == 1.5

        t2D = 0.5, 0.5
        v2D = 1.0e0, 2.0e0, 1.0e0, 2.0e0
        @test lerp(v2D, t2D) == 1.5

        t3D = 0.5, 0.5, 0.5
        v3D = 1.0e0, 2.0e0, 1.0e0, 2.0e0, 1.0e0, 2.0e0, 1.0e0, 2.0e0
        @test lerp(v3D, t3D) == 1.5
    end
end

@testset "Interpolations 2D" begin
    nxcell, max_xcell, min_xcell = 5, 5, 1
    n = 5 # number of vertices
    nx = ny = n - 1
    Lx = Ly = 1.0
    # nodal vertices
    xvi = xv, yv = LinRange(0, Lx, n), LinRange(0, Ly, n)
    dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
    # nodal centers
    xci = xc, yc = LinRange(0 + dx / 2, Lx - dx / 2, n - 1), LinRange(0 + dy / 2, Ly - dy / 2, n - 1)
    # staggered grid velocity nodal locations
    grid_vx = xv, JustPIC._2D.add_periodic_ghost_nodes(yc)
    grid_vy = JustPIC._2D.add_periodic_ghost_nodes(xc), yv

    # Initialize particles & particle fields
    particles = _2D.init_particles(
        backend, nxcell, max_xcell, min_xcell, grid_vx, grid_vy,
    )
    pT, = _2D.init_cell_arrays(particles, Val(1))
    xvi_p = JustPIC._2D.add_periodic_ghost_nodes.(xvi)
    xci_p = JustPIC._2D.add_periodic_ghost_nodes.(xci)

    # Linear field at the vertices
    T = TA(backend)([y for x in xvi_p[1], y in xvi_p[2]])
    T0 = TA(backend)([y for x in xvi_p[1], y in xvi_p[2]])
    # Linear field at the centroids
    Tc = TA(backend)([y for x in xci_p[1], y in xci_p[2]])

    # Grid to particle test
    _2D.grid2particle!(pT, xvi_p, T, particles, diff.(xvi_p))

    active = Array(particles.index.data)
    @test Array(pT.data)[active] ≈ Array(particles.coords[2].data)[active]

    # Grid to particle test
    _2D.grid2particle_flip!(pT, xvi_p, T, T0, particles)

    @test Array(pT.data)[active] ≈ Array(particles.coords[2].data)[active]

    # Particle to grid test
    T2 = similar(T)
    _2D.particle2grid!(T2, pT, particles)
    # norm(T2 .- T) / length(T)
    finite_mask = isfinite.(T2)
    @test norm(T2[finite_mask] .- T[finite_mask]) / count(finite_mask) < 1.0e-1

    # Grid to centroid test
    _2D.centroid2particle!(pT, xci_p, Tc, particles, diff.(xci_p))

    @test Array(pT.data)[active] ≈ Array(particles.coords[2].data)[active]

    # Particle to centroid test
    Tc2 = similar(Tc)
    _2D.particle2centroid!(Tc2, pT, xci_p, particles, diff.(xci_p))
    # norm(T2 .- T) / length(T)
    finite_mask_c = isfinite.(Tc2)
    @test norm(Tc2[finite_mask_c] .- Tc[finite_mask_c]) / count(finite_mask_c) < 1.0e-1

    # test copy function
    particles_copy = copy(particles)
    pT_copy = copy(pT)
    @test particles_copy.index.data[:] == particles.index.data[:]
    @test pT_copy.data[:] == pT.data[:]
end

@testset "Interpolations 3D" begin

    import JustPIC._3D as JP3

    nxcell, max_xcell, min_xcell = 12, 12, 1
    n = 5 # number of vertices
    nx = ny = nz = n - 1
    ni = nx, ny, nz
    Lx = Ly = Lz = 1.0
    Li = Lx, Ly, Lz
    # nodal vertices
    xvi = xv, yv, zv = ntuple(i -> LinRange(0, Li[i], n), Val(3))
    # grid spacing
    dxi = dx, dy, dz = ntuple(i -> xvi[i][2] - xvi[i][1], Val(3))
    # nodal centers
    xci = xc, yc, zc = ntuple(i -> LinRange(0 + dxi[i] / 2, Li[i] - dxi[i] / 2, ni[i]), Val(3))

    # staggered grid velocity nodal locations
    grid_vx = xv, JustPIC._3D.add_periodic_ghost_nodes(yc), JustPIC._3D.add_periodic_ghost_nodes(zc)
    grid_vy = JustPIC._3D.add_periodic_ghost_nodes(xc), yv, JustPIC._3D.add_periodic_ghost_nodes(zc)
    grid_vz = JustPIC._3D.add_periodic_ghost_nodes(xc), JustPIC._3D.add_periodic_ghost_nodes(yc), zv

    # Initialize particles -------------------------------
    particles = JP3.init_particles(
        backend, nxcell, max_xcell, min_xcell, grid_vx, grid_vy, grid_vz
    )
    pT, = JP3.init_cell_arrays(particles, Val(1))
    xvi_p = JustPIC._3D.add_periodic_ghost_nodes.(xvi)
    xci_p = JustPIC._3D.add_periodic_ghost_nodes.(xci)

    # Linear field at the vertices
    T = TA(backend)([z for x in xvi_p[1], y in xvi_p[2], z in xvi_p[3]])
    T0 = TA(backend)([z for x in xvi_p[1], y in xvi_p[2], z in xvi_p[3]])
    # Linear field at the centroids
    Tc = TA(backend)([z for x in xci_p[1], y in xci_p[2], z in xci_p[3]])

    # Grid to particle test
    JP3.grid2particle!(pT, xvi_p, T, particles, diff.(xvi_p))

    active = Array(particles.index.data)
    @test Array(pT.data)[active] ≈ Array(particles.coords[3].data)[active]

    # Grid to particle test
    JP3.grid2particle_flip!(pT, xvi_p, T, T0, particles)

    @test Array(pT.data)[active] ≈ Array(particles.coords[3].data)[active]

    # Particle to grid test
    T2 = similar(T)
    JP3.particle2grid!(T2, pT, particles)
    finite_mask = isfinite.(T2)
    @test norm(T2[finite_mask] .- T[finite_mask]) / count(finite_mask) < 1.0e-1

    # Grid to centroid test
    JP3.centroid2particle!(pT, xci_p, Tc, particles, diff.(xci_p))
    @test Array(pT.data)[active] ≈ Array(particles.coords[3].data)[active]

    # Particle to centroid test
    Tc2 = similar(Tc)
    JP3.particle2centroid!(Tc2, pT, xci_p, particles, diff.(xci_p))
    # norm(T2 .- T) / length(T)
    finite_mask_c = isfinite.(Tc2)
    @test norm(Tc2[finite_mask_c] .- Tc[finite_mask_c]) / count(finite_mask_c) < 1.0e-1

    # test copy function
    particles_copy = copy(particles)
    pT_copy = copy(pT)
    @test particles_copy.index.data[:] == particles.index.data[:]
    @test pT_copy.data[:] == pT.data[:]
end

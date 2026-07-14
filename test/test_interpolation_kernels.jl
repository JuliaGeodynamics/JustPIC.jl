const BACKEND_NAME = get(ENV, "JULIA_JUSTPIC_BACKEND", "CPU")

@static if BACKEND_NAME == "AMDGPU"
    using AMDGPU
    AMDGPU.allowscalar(true)
elseif BACKEND_NAME == "CUDA"
    using CUDA
    CUDA.allowscalar(true)
elseif BACKEND_NAME == "Metal"
    using Metal
    Metal.allowscalar(true)
end

using Test
using JustPIC
using LinearAlgebra
import JustPIC: lerp
import KernelAbstractions: CPU

const backend = @static if BACKEND_NAME == "AMDGPU"
    AMDGPU.ROCBackend
elseif BACKEND_NAME == "CUDA"
    CUDA.CUDABackend
elseif BACKEND_NAME == "Metal"
    Metal.MetalBackend
else
    CPU
end

# Metal has no Float64; JULIA_JUSTPIC_PRECISION=Float32 runs the same paths on CPU
const FT = if BACKEND_NAME == "Metal" || get(ENV, "JULIA_JUSTPIC_PRECISION", "") == "Float32"
    Float32
else
    Float64
end

function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = x1 - dx
    xF = x2 + dx
    return LinRange(xI, xF, n + 2)
end

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
    Lx = Ly = FT(1)
    # nodal vertices
    xvi = xv, yv = LinRange(0, Lx, n), LinRange(0, Ly, n)
    dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
    # nodal centers
    xci = xc, yc = LinRange(0 + dx / 2, Lx - dx / 2, n - 1), LinRange(0 + dy / 2, Ly - dy / 2, n - 1)
    # staggered grid velocity nodal locations
    grid_vx = xv, expand_range(yc)
    grid_vy = expand_range(xc), yv
    xvi_device = TA(backend).(xvi)
    grid_vel = TA(backend).(grid_vx), TA(backend).(grid_vy)
    grid_vx = xv, JustPIC.add_periodic_ghost_nodes(yc)
    grid_vy = JustPIC.add_periodic_ghost_nodes(xc), yv

    # Initialize particles & particle fields
    particles = JustPIC.init_particles(
        backend, nxcell, max_xcell, min_xcell, grid_vel...,
    )
    pT, = JustPIC.init_cell_arrays(particles, Val(1))
    xvi_p = JustPIC.add_periodic_ghost_nodes.(xvi)
    xci_p = JustPIC.add_periodic_ghost_nodes.(xci)

    # Linear field at the vertices
    T = TA(backend)([y for x in xvi_p[1], y in xvi_p[2]])
    T0 = TA(backend)([y for x in xvi_p[1], y in xvi_p[2]])
    # Linear field at the centroids
    Tc = TA(backend)([y for x in xci_p[1], y in xci_p[2]])

    # Grid to particle test
    JustPIC.grid2particle!(pT, xvi_p, T, particles, diff.(xvi_p))

    active = Array(particles.index.data)
    @test Array(pT.data)[active] ≈ Array(particles.coords[2].data)[active]

    # Grid to particle test
    JustPIC.grid2particle_flip!(pT, xvi_p, T, T0, particles)

    @test Array(pT.data)[active] ≈ Array(particles.coords[2].data)[active]

    # Particle to grid test
    T2 = similar(T)
    fill!(T2, NaN)
    JustPIC.particle2grid!(T2, pT, particles)
    # norm(T2 .- T) / length(T)
    finite_mask = isfinite.(T2)
    @test norm(T2[finite_mask] .- T[finite_mask]) / count(finite_mask) < 1.0e-1

    # Grid to centroid test
    JustPIC.centroid2particle!(pT, xci_p, Tc, particles, diff.(xci_p))

    @test Array(pT.data)[active] ≈ Array(particles.coords[2].data)[active]

    # Particle to centroid test
    Tc2 = similar(Tc)
    JustPIC.particle2centroid!(Tc2, pT, particles)
    fill!(Tc2, NaN)
    JustPIC.particle2centroid!(Tc2, pT, xci_p, particles, diff.(xci_p))
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


    nxcell, max_xcell, min_xcell = 12, 12, 1
    n = 5 # number of vertices
    nx = ny = nz = n - 1
    ni = nx, ny, nz
    Lx = Ly = Lz = FT(1)
    Li = Lx, Ly, Lz
    # nodal vertices
    xvi = xv, yv, zv = ntuple(i -> LinRange(0, Li[i], n), Val(3))
    # grid spacing
    dxi = dx, dy, dz = ntuple(i -> xvi[i][2] - xvi[i][1], Val(3))
    # nodal centers
    xci = xc, yc, zc = ntuple(i -> LinRange(0 + dxi[i] / 2, Li[i] - dxi[i] / 2, ni[i]), Val(3))
    grid_vx = xv, expand_range(yc), expand_range(zc)
    grid_vy = expand_range(xc), yv, expand_range(zc)
    grid_vz = expand_range(xc), expand_range(yc), zv
    xvi_device = TA(backend).(xvi)
    grid_vel = TA(backend).(grid_vx), TA(backend).(grid_vy), TA(backend).(grid_vz)

    # staggered grid velocity nodal locations
    grid_vx = xv, JustPIC.add_periodic_ghost_nodes(yc), JustPIC.add_periodic_ghost_nodes(zc)
    grid_vy = JustPIC.add_periodic_ghost_nodes(xc), yv, JustPIC.add_periodic_ghost_nodes(zc)
    grid_vz = JustPIC.add_periodic_ghost_nodes(xc), JustPIC.add_periodic_ghost_nodes(yc), zv

    # Initialize particles -------------------------------
    particles = JustPIC.init_particles(
        backend, nxcell, max_xcell, min_xcell, grid_vel...
    )
    pT, = JustPIC.init_cell_arrays(particles, Val(1))
    xvi_p = JustPIC.add_periodic_ghost_nodes.(xvi)
    xci_p = JustPIC.add_periodic_ghost_nodes.(xci)

    # Linear field at the vertices
    T = TA(backend)([z for x in xvi_p[1], y in xvi_p[2], z in xvi_p[3]])
    T0 = TA(backend)([z for x in xvi_p[1], y in xvi_p[2], z in xvi_p[3]])
    # Linear field at the centroids
    Tc = TA(backend)([z for x in xci_p[1], y in xci_p[2], z in xci_p[3]])

    # Grid to particle test
    JustPIC.grid2particle!(pT, xvi_p, T, particles, diff.(xvi_p))

    active = Array(particles.index.data)
    @test Array(pT.data)[active] ≈ Array(particles.coords[3].data)[active]

    # Grid to particle test
    JustPIC.grid2particle_flip!(pT, xvi_p, T, T0, particles)

    @test Array(pT.data)[active] ≈ Array(particles.coords[3].data)[active]

    # Particle to grid test
    T2 = similar(T)
    fill!(T2, NaN)
    JustPIC.particle2grid!(T2, pT, particles)
    finite_mask = isfinite.(T2)
    @test norm(T2[finite_mask] .- T[finite_mask]) / count(finite_mask) < 1.0e-1

    # Grid to centroid test
    JustPIC.centroid2particle!(pT, xci_p, Tc, particles, diff.(xci_p))
    @test Array(pT.data)[active] ≈ Array(particles.coords[3].data)[active]

    # Particle to centroid test
    Tc2 = similar(Tc)
    fill!(Tc2, NaN)
    JustPIC.particle2centroid!(Tc2, pT, xci_p, particles, diff.(xci_p))
    # norm(T2 .- T) / length(T)
    finite_mask_c = isfinite.(Tc2)
    @test norm(Tc2[finite_mask_c] .- Tc[finite_mask_c]) / count(finite_mask_c) < 1.0e-1

    # test copy function
    particles_copy = copy(particles)
    pT_copy = copy(pT)
    @test particles_copy.index.data[:] == particles.index.data[:]
    @test pT_copy.data[:] == pT.data[:]
end

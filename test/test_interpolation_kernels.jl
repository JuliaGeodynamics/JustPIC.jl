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

include(joinpath(@__DIR__, "helpers_backend.jl"))
check_backend(BACKEND_NAME, backend, FT)

function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = x1 - dx
    xF = x2 + dx
    return LinRange(xI, xF, n + 2)
end

function interior_support(A)
    mask = falses(size(A))
    ranges = ntuple(i -> 2:(size(A, i) - 1), ndims(A))
    mask[ranges...] .= true
    return mask
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

    pX, pY = JustPIC.init_cell_arrays(particles, Val(2))
    X = TA(backend)([x for x in xvi_p[1], y in xvi_p[2]])
    JustPIC.grid2particle!((pX, pY), (X, T), particles)
    @test Array(pX.data)[active] ≈ Array(particles.coords[1].data)[active]
    @test Array(pY.data)[active] ≈ Array(particles.coords[2].data)[active]

    # Grid to particle test
    JustPIC.grid2particle_flip!(pT, xvi_p, T, T0, particles)

    @test Array(pT.data)[active] ≈ Array(particles.coords[2].data)[active]

    # Particle to grid test
    T2 = similar(T)
    fill!(T2, eltype(T2)(NaN))
    JustPIC.particle2grid!(T2, pT, particles)
    # norm(T2 .- T) / length(T)
    support = interior_support(T2)
    @test all(isfinite.(T2[support]))
    @test all(isnan.(T2[.!support]))
    @test norm(T2[support] .- T[support]) / count(support) < 1.0e-1

    # Grid to centroid test
    JustPIC.centroid2particle!(pT, xci_p, Tc, particles, diff.(xci_p))

    @test Array(pT.data)[active] ≈ Array(particles.coords[2].data)[active]

    # Particle to centroid test
    Tc2 = similar(Tc)
    JustPIC.particle2centroid!(Tc2, pT, particles)
    fill!(Tc2, eltype(Tc2)(NaN))
    JustPIC.particle2centroid!(Tc2, pT, xci_p, particles, diff.(xci_p))
    # norm(T2 .- T) / length(T)
    support_c = interior_support(Tc2)
    @test all(isfinite.(Tc2[support_c]))
    @test all(isnan.(Tc2[.!support_c]))
    @test norm(Tc2[support_c] .- Tc[support_c]) / count(support_c) < 1.0e-1

    # test copy function
    particles_copy = copy(particles)
    pT_copy = copy(pT)
    @test particles_copy.index.data[:] == particles.index.data[:]
    @test pT_copy.data[:] == pT.data[:]
end

@testset "Ghost-node opt-out 2D" begin
    nxcell, max_xcell, min_xcell = 5, 5, 1
    n = 5 # number of vertices
    Lx = Ly = FT(1)
    xvi = xv, yv = LinRange(0, Lx, n), LinRange(0, Ly, n)
    dx, dy = xv[2] - xv[1], yv[2] - yv[1]
    xci = xc, yc = LinRange(0 + dx / 2, Lx - dx / 2, n - 1), LinRange(0 + dy / 2, Ly - dy / 2, n - 1)
    grid_vel = TA(backend).((xv, expand_range(yc))), TA(backend).((expand_range(xc), yv))

    particles = JustPIC.init_particles(
        backend, nxcell, max_xcell, min_xcell, grid_vel...,
    )
    xvi_p = JustPIC.add_periodic_ghost_nodes.(xvi)
    xci_p = JustPIC.add_periodic_ghost_nodes.(xci)

    pT, pX = JustPIC.init_cell_arrays(particles, Val(2))
    JustPIC.grid2particle!(pT, TA(backend)([y for x in xvi_p[1], y in xvi_p[2]]), particles)
    JustPIC.grid2particle!(pX, TA(backend)([x for x in xvi_p[1], y in xvi_p[2]]), particles)

    # a tuple of destination fields must agree with the same fields interpolated one by one
    ghost_nan() = TA(backend)(fill(FT(NaN), length.(xvi_p)))
    interior(A) = Array(A)[2:(end - 1), 2:(end - 1)]
    Fx, Fy = ghost_nan(), ghost_nan()
    Fx_ref, Fy_ref = ghost_nan(), ghost_nan()
    JustPIC.particle2grid!((Fx, Fy), (pX, pT), particles)
    JustPIC.particle2grid!(Fx_ref, pX, particles)
    JustPIC.particle2grid!(Fy_ref, pT, particles)
    @test interior(Fx) ≈ interior(Fx_ref)
    @test interior(Fy) ≈ interior(Fy_ref)

    # `ghost_i = false` writes the physical nodes of an unghosted array
    Fv_plain = TA(backend)(fill(FT(NaN), length.(xvi)))
    JustPIC.particle2grid!(Fv_plain, pT, particles; ghost_1 = false, ghost_2 = false)
    @test Array(Fv_plain) == interior(Fy_ref)

    # Empty particle neighborhoods must produce zero rather than NaN from 0 / 0.
    particles_empty = copy(particles)
    fill!(particles_empty.index.data, false)
    Fempty = TA(backend)(fill(FT(NaN), length.(xvi)))
    JustPIC.particle2grid!(Fempty, pT, particles_empty; ghost_1 = false, ghost_2 = false)
    @test all(iszero, Fempty)

    Fc = TA(backend)(fill(FT(NaN), length.(xci_p)))
    Fc_plain = TA(backend)(fill(FT(NaN), length.(xci)))
    JustPIC.particle2centroid!(Fc, pT, particles)
    JustPIC.particle2centroid!(Fc_plain, pT, particles; ghost_1 = false, ghost_2 = false)
    @test Array(Fc_plain) == Array(Fc)[2:(end - 1), 2:(end - 1)]

    # the same field read back with and without ghost nodes must reach the particles alike
    T_ghost = TA(backend)([y for x in xvi_p[1], y in xvi_p[2]])
    T_plain = TA(backend)([y for x in xvi[1], y in xvi[2]])
    pT_ghost, pT_plain = JustPIC.init_cell_arrays(particles, Val(2))
    JustPIC.grid2particle!(pT_ghost, T_ghost, particles)
    JustPIC.grid2particle!(pT_plain, T_plain, particles; ghost_1 = false, ghost_2 = false)
    active = Array(particles.index.data)
    @test Array(pT_ghost.data)[active] ≈ Array(pT_plain.data)[active]

    pF_ghost, pF_plain = JustPIC.init_cell_arrays(particles, Val(2))
    JustPIC.grid2particle_flip!(pF_ghost, xvi_p, T_ghost, T_ghost, particles; α = FT(0.5))
    JustPIC.grid2particle_flip!(
        pF_plain, xvi_p, T_plain, T_plain, particles;
        α = FT(0.5), ghost_1 = false, ghost_2 = false,
    )
    @test Array(pF_ghost.data)[active] ≈ Array(pF_plain.data)[active]

    Tc_plain = TA(backend)([y for x in xci[1], y in xci[2]])
    pTc_plain, = JustPIC.init_cell_arrays(particles, Val(1))
    JustPIC.centroid2particle!(pTc_plain, Tc_plain, particles; ghosted = false)
    @test Array(pTc_plain.data)[active] ≈ Array(particles.coords[2].data)[active]
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

    pX, pZ = JustPIC.init_cell_arrays(particles, Val(2))
    X = TA(backend)([x for x in xvi_p[1], y in xvi_p[2], z in xvi_p[3]])
    JustPIC.grid2particle!((pX, pZ), (X, T), particles)
    @test Array(pX.data)[active] ≈ Array(particles.coords[1].data)[active]
    @test Array(pZ.data)[active] ≈ Array(particles.coords[3].data)[active]

    # Grid to particle test
    JustPIC.grid2particle_flip!(pT, xvi_p, T, T0, particles)

    @test Array(pT.data)[active] ≈ Array(particles.coords[3].data)[active]

    # Particle to grid test
    T2 = similar(T)
    fill!(T2, eltype(T2)(NaN))
    JustPIC.particle2grid!(T2, pT, particles)
    support = interior_support(T2)
    @test all(isfinite.(T2[support]))
    @test all(isnan.(T2[.!support]))
    @test norm(T2[support] .- T[support]) / count(support) < 1.0e-1

    # Grid to centroid test
    JustPIC.centroid2particle!(pT, xci_p, Tc, particles, diff.(xci_p))
    @test Array(pT.data)[active] ≈ Array(particles.coords[3].data)[active]

    # Particle to centroid test
    Tc2 = similar(Tc)
    fill!(Tc2, eltype(Tc2)(NaN))
    JustPIC.particle2centroid!(Tc2, pT, xci_p, particles, diff.(xci_p))
    # norm(T2 .- T) / length(T)
    support_c = interior_support(Tc2)
    @test all(isfinite.(Tc2[support_c]))
    @test all(isnan.(Tc2[.!support_c]))
    @test norm(Tc2[support_c] .- Tc[support_c]) / count(support_c) < 1.0e-1

    # test copy function
    particles_copy = copy(particles)
    pT_copy = copy(pT)
    @test particles_copy.index.data[:] == particles.index.data[:]
    @test pT_copy.data[:] == pT.data[:]
end

@testset "Refined-grid PIC/FLIP equivalence" begin
    if BACKEND_NAME == "CPU"

        xv = FT[0, 0.1, 0.3, 0.7, 1]
        yv = FT[0, 0.2, 0.5, 0.8, 1]
        xc = (xv[1:(end - 1)] .+ xv[2:end]) ./ 2
        yc = (yv[1:(end - 1)] .+ yv[2:end]) ./ 2
        grid_vx = xv, JustPIC.add_periodic_ghost_nodes(yc)
        grid_vy = JustPIC.add_periodic_ghost_nodes(xc), yv
        particles = JustPIC.init_particles(CPU, 4, 4, 1, grid_vx, grid_vy)
        p_pic, = JustPIC.init_cell_arrays(particles, Val(1))
        p_flip, = JustPIC.init_cell_arrays(particles, Val(1))
        xvi = particles.xvi
        T = [x + y for x in xvi[1], y in xvi[2]]

        JustPIC.grid2particle!(p_pic, T, particles)
        JustPIC.grid2particle_flip!(p_flip, xvi, T, T, particles; α = FT(1))

        @test p_flip.data == p_pic.data
    end
end

@testset "Passive markers on refined grids" begin
    if BACKEND_NAME == "CPU"
        xv = FT[0, 0.1, 0.3, 0.7, 1]
        yv = FT[0, 0.2, 0.5, 0.8, 1]
        markers = JustPIC.init_passive_markers(CPU, (FT[0.15, 0.65], FT[0.25, 0.7]))
        field = [x + y for x in xv, y in yv]
        values = similar(markers.coords[1])

        JustPIC.grid2particle!(values, (xv, yv), field, markers)

        @test values ≈ markers.coords[1] .+ markers.coords[2]
    end
end

@testset "3D MQS stencil consistency" begin
    F = [FT(i + 10j + 100k) for i in 1:6, j in 1:6, k in 1:6]
    idx = (2, 2, 2)
    t = (FT(0.3), FT(0.4), FT(0.6))
    v = JustPIC.field_corners(F, idx)

    bottom = JustPIC.MQS(view(F, :, :, 2), v[1:4], t[1:2], 2, 2, Val(1))
    top = JustPIC.MQS(view(F, :, :, 3), v[5:8], t[1:2], 2, 2, Val(1))
    expected_x = lerp((bottom, top), (t[3],))
    @test JustPIC.MQS(F, v, t, idx..., Val(1)) ≈ expected_x

    F_y = [FT(j) for i in 1:6, j in 1:6, k in 1:6]
    v_y = JustPIC.field_corners(F_y, idx)
    @test JustPIC.MQS(F_y, v_y, t, idx..., Val(3)) ≈ FT(2) + t[2]

    F_z = [FT(k)^2 for i in 1:6, j in 1:6, k in 1:6]
    v_z = JustPIC.field_corners(F_z, idx)
    F_xz = F_z[:, 2, :]
    expected_z = JustPIC.MQS(
        F_xz, (v_z[1], v_z[2], v_z[5], v_z[6]), (t[1], t[3]), 2, 2, Val(2)
    )
    @test JustPIC.MQS(F_z, v_z, t, idx..., Val(3)) ≈ expected_z
end

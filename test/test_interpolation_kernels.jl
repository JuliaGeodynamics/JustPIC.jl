const BACKEND_NAME = get(ENV, "JULIA_JUSTPIC_BACKEND", "CPU")

@static if BACKEND_NAME == "AMDGPU"
    using AMDGPU
elseif BACKEND_NAME == "CUDA"
    using CUDA
elseif BACKEND_NAME == "Metal"
    using Metal
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
    JustPIC.grid2particle_flip!(pT, T, T0, particles)

    @test Array(pT.data)[active] ≈ Array(particles.coords[2].data)[active]

    # Particle to grid test
    T2 = similar(T)
    fill!(T2, eltype(T2)(NaN))
    JustPIC.particle2grid!(T2, pT, particles)
    T2 = Array(T2)
    T = Array(T)
    # norm(T2 .- T) / length(T)
    support = interior_support(T2)
    @test all(isfinite.(T2[support]))
    @test all(isnan.(T2[.!support]))
    @test norm(T2[support] .- T[support]) / count(support) < 1.0e-1

    # Grid to centroid test
    JustPIC.centroid2particle!(pT, xci_p, Tc, particles, diff.(xci_p))

    @test Array(pT.data)[active] ≈ Array(particles.coords[2].data)[active]

    pT_tuple = JustPIC.init_cell_arrays(particles, Val(3))
    Tc_tuple = ntuple(_ -> TA(backend)(Array(Tc)), Val(3))
    JustPIC.centroid2particle!(pT_tuple, xci_p, Tc_tuple, particles, diff.(xci_p))
    for i in 2:3
        @test Array(pT_tuple[i].data)[active] ≈ Array(pT_tuple[1].data)[active]
    end

    # Particle to centroid test
    Tc2 = similar(Tc)
    JustPIC.particle2centroid!(Tc2, pT, particles)
    fill!(Tc2, eltype(Tc2)(NaN))
    JustPIC.particle2centroid!(Tc2, pT, xci_p, particles, diff.(xci_p))
    Tc2 = Array(Tc2)
    Tc = Array(Tc)
    # norm(T2 .- T) / length(T)
    support_c = interior_support(Tc2)
    @test all(isfinite.(Tc2[support_c]))
    @test all(isnan.(Tc2[.!support_c]))
    @test norm(Tc2[support_c] .- Tc[support_c]) / count(support_c) < 1.0e-1

    # Tuple particle to centroid: weighted averages of constant fields are exact
    pT_tuple = JustPIC.init_cell_arrays(particles, Val(3))
    for i in 1:3
        fill!(pT_tuple[i].data, i)
    end
    Tc_tuple = ntuple(_ -> TA(backend)(fill(FT(NaN), size(Tc2))), Val(3))
    JustPIC.particle2centroid!(Tc_tuple, pT_tuple, particles)
    for i in 1:3
        @test Array(Tc_tuple[i])[support_c] ≈ fill(FT(i), count(support_c))
    end

    # test copy function
    particles_copy = copy(particles)
    @test particles_copy.index.data[:] == particles.index.data[:]
    @test isequal(Array(particles_copy.coords[1].data), Array(particles.coords[1].data))
    @test particles_copy.coords[1].data !== particles.coords[1].data
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

    # Nodes without particle support are NaN.
    particles_empty = copy(particles)
    fill!(particles_empty.index.data, false)
    Fempty = TA(backend)(zeros(FT, length.(xvi)))
    JustPIC.particle2grid!(Fempty, pT, particles_empty; ghost_1 = false, ghost_2 = false)
    @test all(isnan, Array(Fempty))

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
    JustPIC.grid2particle_flip!(pF_ghost, T_ghost, T_ghost, particles; α = FT(0.5))
    JustPIC.grid2particle_flip!(
        pF_plain, T_plain, T_plain, particles;
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
    JustPIC.grid2particle_flip!(pT, T, T0, particles)

    @test Array(pT.data)[active] ≈ Array(particles.coords[3].data)[active]

    # Particle to grid test
    T2 = similar(T)
    fill!(T2, eltype(T2)(NaN))
    JustPIC.particle2grid!(T2, pT, particles)
    T2 = Array(T2)
    T = Array(T)
    support = interior_support(T2)
    @test all(isfinite.(T2[support]))
    @test all(isnan.(T2[.!support]))
    @test norm(T2[support] .- T[support]) / count(support) < 1.0e-1

    # Grid to centroid test
    JustPIC.centroid2particle!(pT, xci_p, Tc, particles, diff.(xci_p))
    @test Array(pT.data)[active] ≈ Array(particles.coords[3].data)[active]

    pT_tuple = JustPIC.init_cell_arrays(particles, Val(3))
    Tc_tuple = ntuple(_ -> TA(backend)(Array(Tc)), Val(3))
    JustPIC.centroid2particle!(pT_tuple, xci_p, Tc_tuple, particles, diff.(xci_p))
    for i in 2:3
        @test Array(pT_tuple[i].data)[active] ≈ Array(pT_tuple[1].data)[active]
    end

    # Particle to centroid test
    Tc2 = similar(Tc)
    fill!(Tc2, eltype(Tc2)(NaN))
    JustPIC.particle2centroid!(Tc2, pT, xci_p, particles, diff.(xci_p))
    Tc2 = Array(Tc2)
    Tc = Array(Tc)
    # norm(T2 .- T) / length(T)
    support_c = interior_support(Tc2)
    @test all(isfinite.(Tc2[support_c]))
    @test all(isnan.(Tc2[.!support_c]))
    @test norm(Tc2[support_c] .- Tc[support_c]) / count(support_c) < 1.0e-1

    # Tuple particle to centroid: weighted averages of constant fields are exact
    pT_tuple = JustPIC.init_cell_arrays(particles, Val(3))
    for i in 1:3
        fill!(pT_tuple[i].data, i)
    end
    Tc_tuple = ntuple(_ -> TA(backend)(fill(FT(NaN), size(Tc2))), Val(3))
    JustPIC.particle2centroid!(Tc_tuple, pT_tuple, particles)
    for i in 1:3
        @test Array(Tc_tuple[i])[support_c] ≈ fill(FT(i), count(support_c))
    end

    # test copy function
    particles_copy = copy(particles)
    @test particles_copy.index.data[:] == particles.index.data[:]
    @test isequal(Array(particles_copy.coords[1].data), Array(particles.coords[1].data))
    @test particles_copy.coords[1].data !== particles.coords[1].data
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
        JustPIC.grid2particle_flip!(p_flip, T, T, particles; α = FT(1))

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

@testset "Passive markers 3D scatter" begin
    if BACKEND_NAME == "CPU"
        xv = FT[0, 0.1, 0.3, 0.7, 1]
        yv = FT[0, 0.2, 0.5, 0.8, 1]
        zv = FT[0, 0.15, 0.4, 0.75, 1]
        coords = (FT[0.1, 0.3, 0.7, 1], FT[0.2, 0.5, 0.8, 1], FT[0.15, 0.4, 0.75, 1])
        markers = JustPIC.init_passive_markers(CPU, coords)
        field = [x + 2y + 3z for x in xv, y in yv, z in zv]
        values = similar(coords[1])
        output = zeros(FT, length(xv), length(yv), length(zv))
        buffer = similar(output)

        JustPIC.grid2particle!(values, (xv, yv, zv), field, markers)
        JustPIC.particle2grid!(output, values, buffer, (xv, yv, zv), markers)

        # support: nodes that receive a positive bilinear weight from some marker. Corners
        # of a marker's cell with zero weight may round either way, so only nodes outside
        # every marker's cell are required to be NaN.
        grid = (xv, yv, zv)
        weight = zeros(FT, size(output))
        touched = falses(size(output))
        for k in eachindex(coords[1])
            p = getindex.(coords, k)
            I = ntuple(d -> clamp(searchsortedlast(grid[d], p[d]), 1, length(grid[d]) - 1), 3)
            for offset in CartesianIndices((0:1, 0:1, 0:1))
                node = I .+ Tuple(offset)
                w = prod(d -> 1 - abs(p[d] - grid[d][node[d]]) / (grid[d][I[d] + 1] - grid[d][I[d]]), 1:3)
                weight[node...] = max(weight[node...], w)
                touched[node...] = true
            end
        end
        support = weight .> sqrt(eps(FT))
        @test all(isfinite, output[support])
        @test all(isnan, output[.!touched])
        @test output[2, 2, 2] ≈ 0.1 + 2 * 0.2 + 3 * 0.15
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

@testset "PIC/FLIP grid to particle 2D" begin
    n = 5
    xv = yv = LinRange(FT(0), FT(1), n)
    dx = xv[2] - xv[1]
    xc = LinRange(dx / 2, 1 - dx / 2, n - 1)
    xvi = xv, yv
    grid_vx = TA(backend).((xv, expand_range(xc)))
    grid_vy = TA(backend).((expand_range(xc), yv))
    particles = JustPIC.init_particles(backend, 5, 5, 1, grid_vx, grid_vy)
    active = Array(particles.index.data)
    xp = Array(particles.coords[1].data)[active]
    yp = Array(particles.coords[2].data)[active]
    xg, yg = particles.xvi

    # linear fields: bilinear interpolation reproduces them exactly
    Y = TA(backend)(FT[y for x in xg, y in yg])
    X = TA(backend)(FT[x for x in xg, y in yg])
    ΔX = FT(0.1) .* X
    Y1 = Y .+ ΔX
    c = FT(0.1)
    tol = 50 * eps(FT)

    # particle values that are *not* consistent with the old grid field, so that PIC and
    # FLIP give different answers: pT = 2y
    function fresh_particle_field()
        pT, = JustPIC.init_cell_arrays(particles, Val(1))
        JustPIC.grid2particle!(pT, 2 .* Y, particles)
        return pT
    end

    @testset "α = 1 is pure PIC" begin
        pT = fresh_particle_field()
        grid2particle_flip!(pT, Y1, Y, particles; α = FT(1))
        @test isapprox(Array(pT.data)[active], yp .+ c .* xp; atol = tol)
    end

    @testset "α = 0 is pure FLIP" begin
        pT = fresh_particle_field()
        grid2particle_flip!(pT, Y1, Y, particles; α = FT(0))
        @test isapprox(Array(pT.data)[active], 2 .* yp .+ c .* xp; atol = tol)
    end

    @testset "α = 1/2 blends" begin
        pT = fresh_particle_field()
        grid2particle_flip!(pT, Y1, Y, particles; α = FT(0.5))
        @test isapprox(Array(pT.data)[active], FT(1.5) .* yp .+ c .* xp; atol = tol)
    end

    @testset "unchanged grid leaves FLIP particles untouched" begin
        pT = fresh_particle_field()
        before = copy(Array(pT.data))
        grid2particle_flip!(pT, Y, Y .+ 0, particles; α = FT(0))
        @test Array(pT.data)[active] == before[active]
    end

    @testset "tuple of fields" begin
        pA = fresh_particle_field()
        pB, = JustPIC.init_cell_arrays(particles, Val(1))
        JustPIC.grid2particle!(pB, 3 .* X, particles)
        grid2particle_flip!(
            (pA, pB), (Y1, X .+ Y), (Y, X), particles; α = FT(0)
        )
        @test isapprox(Array(pA.data)[active], 2 .* yp .+ c .* xp; atol = tol)
        @test isapprox(Array(pB.data)[active], 3 .* xp .+ yp; atol = tol)
    end

    @testset "deprecated explicit-grid method still works" begin
        pNew = fresh_particle_field()
        pOld = fresh_particle_field()
        grid2particle_flip!(pNew, Y1, Y, particles; α = FT(0.5))
        grid2particle_flip!(pOld, particles.xvi, Y1, Y, particles; α = FT(0.5))
        @test Array(pOld.data)[active] == Array(pNew.data)[active]
    end

    @testset "precision is preserved" begin
        pT = fresh_particle_field()
        grid2particle_flip!(pT, Y1, Y, particles; α = FT(0.5))
        @test eltype(eltype(pT)) === FT
    end
end

function flip_test_particles(::Val{D}) where {D}
    n = 5
    xv = LinRange(FT(0), FT(1), n)
    dx = xv[2] - xv[1]
    xc = LinRange(dx / 2, 1 - dx / 2, n - 1)
    grids = ntuple(Val(D)) do d
        TA(backend).(ntuple(k -> k == d ? xv : expand_range(xc), Val(D)))
    end
    nxcell = D == 2 ? 5 : 12
    return JustPIC.init_particles(backend, nxcell, nxcell, 1, grids...)
end

# linear field c ⋅ x sampled on the tuple of coordinate vectors `xs`
linear_field(xs, c) = TA(backend)(FT[sum(c .* x) for x in Iterators.product(map(collect, xs)...)])

@testset "PIC/FLIP transfers $(D)D" for D in (2, 3)
    particles = flip_test_particles(Val(D))
    active = Array(particles.index.data)
    xp = ntuple(d -> Array(particles.coords[d].data)[active], Val(D))
    at_particles(c) = sum(c[d] .* xp[d] for d in 1:D)
    host(pT) = Array(pT.data)[active]
    tol = 200 * eps(FT)
    c0 = ntuple(d -> FT(d), Val(D))
    c1 = ntuple(d -> FT(d) + FT(0.1) * d^2, Val(D))
    # particle field that is inconsistent with the grid fields, so PIC and FLIP differ
    function fresh_particle_field(F)
        pT, = JustPIC.init_cell_arrays(particles, Val(1))
        JustPIC.grid2particle!(pT, 2 .* F, particles)
        return pT
    end
    expected(α) = α .* at_particles(c1) .+ (1 - α) .* (2 .* at_particles(c0) .+ at_particles(c1) .- at_particles(c0))
    full_grids(xs, ghost_1) = ntuple(d -> (d == 1 && !ghost_1) ? xs[d][2:(end - 1)] : xs[d], Val(D))

    @testset "grid2particle_flip!" begin
        F0, F1 = linear_field(particles.xvi, c0), linear_field(particles.xvi, c1)
        for α in FT.((0, 1 // 2, 1))
            pT = fresh_particle_field(F0)
            grid2particle_flip!(pT, F1, F0, particles; α)
            @test isapprox(host(pT), expected(α); atol = tol)
            @test eltype(eltype(pT)) === FT
        end
        pA, pB = fresh_particle_field(F0), fresh_particle_field(F0)
        grid2particle_flip!((pA, pB), (F1, copy(F1)), (F0, copy(F0)), particles; α = FT(0))
        @test isapprox(host(pA), expected(0); atol = tol)
        @test host(pA) == host(pB)
    end

    @testset "centroid2particle_flip!" begin
        for ghosted in (true, false)
            xs = ghosted ? particles.xci : map(x -> x[2:(end - 1)], particles.xci)
            C0, C1 = linear_field(xs, c0), linear_field(xs, c1)
            for α in FT.((0, 1 // 2, 1))
                pT = fresh_particle_field(linear_field(particles.xvi, c0))
                centroid2particle_flip!(pT, C1, C0, particles; α, ghosted)
                @test isapprox(host(pT), expected(α); atol = tol)
                @test eltype(eltype(pT)) === FT
            end
            pA, pB = ntuple(_ -> fresh_particle_field(linear_field(particles.xvi, c0)), Val(2))
            centroid2particle_flip!((pA, pB), (C1, copy(C1)), (C0, copy(C0)), particles; α = FT(0), ghosted)
            @test isapprox(host(pA), expected(0); atol = tol)
            @test host(pA) == host(pB)
        end
    end

    @testset "centroid2particle! ghost layouts" begin
        for ghosted in (true, false)
            xs = ghosted ? particles.xci : map(x -> x[2:(end - 1)], particles.xci)
            pT, pT2 = JustPIC.init_cell_arrays(particles, Val(2))
            centroid2particle!(pT, linear_field(xs, c0), particles; ghosted)
            @test isapprox(host(pT), at_particles(c0); atol = tol)
            centroid2particle!((pT, pT2), (linear_field(xs, c0), linear_field(xs, c1)), particles; ghosted)
            @test isapprox(host(pT), at_particles(c0); atol = tol)
            @test isapprox(host(pT2), at_particles(c1); atol = tol)
        end
        xs = full_grids(particles.xci, false)
        pT, = JustPIC.init_cell_arrays(particles, Val(1))
        centroid2particle!(pT, linear_field(xs, c0), particles; ghost_1 = false)
        @test isapprox(host(pT), at_particles(c0); atol = tol)
    end

    # a constant particle increment shifts every node/centroid that has particles by exactly that amount
    function check_increment(transfer!, xs, extra...)
        δ = FT(0.25)
        F = linear_field(xs, c0)
        Fp0, = JustPIC.init_cell_arrays(particles, Val(1))
        JustPIC.grid2particle!(Fp0, linear_field(particles.xvi, c1), particles)
        Fp, = JustPIC.init_cell_arrays(particles, Val(1))
        Fp.data .= Fp0.data .+ δ
        before = Array(F)
        transfer!(F, Fp, Fp0, particles; extra...)
        @test eltype(F) === FT
        Δ = Array(F) .- before
        @test all(d -> d == 0 || isapprox(d, δ; atol = tol), Δ)
        @test count(d -> d ≠ 0, Δ) ≥ prod(size(F) .- 2)

        # tuple of fields, one of them unchanged
        G, H = linear_field(xs, c0), linear_field(xs, c0)
        Fq, = JustPIC.init_cell_arrays(particles, Val(1))
        Fq.data .= Fp0.data
        Fq0 = deepcopy(Fp0)
        transfer!((G, H), (Fp, Fq), (Fp0, Fq0), particles; extra...)
        @test Array(G) == Array(F)
        @test Array(H) == before

        # no particles, no change
        empty = flip_test_particles(Val(D))
        fill!(empty.index.data, false)
        K = linear_field(xs, c0)
        transfer!(K, Fp, Fp0, empty; extra...)
        @test Array(K) == before
    end

    @testset "particle2grid_flip!" begin
        check_increment(particle2grid_flip!, particles.xvi)
    end

    @testset "particle2centroid_flip!" begin
        check_increment(particle2centroid_flip!, particles.xci)
    end

    @testset "particle2grid_flip! without ghost nodes along x" begin
        Fp0, = JustPIC.init_cell_arrays(particles, Val(1))
        JustPIC.grid2particle!(Fp0, linear_field(particles.xvi, c1), particles)
        Fp, = JustPIC.init_cell_arrays(particles, Val(1))
        Fp.data .= Fp0.data .+ FT(0.25)
        for (transfer!, xs) in ((particle2grid_flip!, particles.xvi), (particle2centroid_flip!, particles.xci))
            Fg = linear_field(xs, c0)
            transfer!(Fg, Fp, Fp0, particles)
            Fl = linear_field(full_grids(xs, false), c0)
            transfer!(Fl, Fp, Fp0, particles; ghost_1 = false)
            @test Array(Fl) == Array(Fg)[2:(end - 1), ntuple(_ -> :, D - 1)...]
        end
    end
end

@testset "particle2centroid! $(D)D" for D in (2, 3)
    particles = flip_test_particles(Val(D))
    c = ntuple(d -> FT(d), Val(D))
    cs = ntuple(d -> FT(2d), Val(D))
    pA, pB = JustPIC.init_cell_arrays(particles, Val(2))
    JustPIC.grid2particle!(pA, linear_field(particles.xvi, c), particles)
    JustPIC.grid2particle!(pB, linear_field(particles.xvi, cs), particles)
    zeros_c() = TA(backend)(zeros(FT, map(length, particles.xci)))

    @testset "tuple matches scalar" begin
        A, B = zeros_c(), zeros_c()
        particle2centroid!(A, pA, particles)
        particle2centroid!(B, pB, particles)
        TA_, TB_ = zeros_c(), zeros_c()
        particle2centroid!((TA_, TB_), (pA, pB), particles)
        @test isapprox(Array(TA_), Array(A); rtol = 100 * eps(FT))
        @test isapprox(Array(TB_), Array(B); rtol = 100 * eps(FT))
    end

    @testset "occupancy mask decides" begin
        masked = flip_test_particles(Val(D))
        fill!(masked.index.data, false)
        C = zeros_c()
        particle2centroid!(C, pA, masked)
        @test all(isnan, Array(C)[map(n -> 2:(n - 1), size(C))...])
    end
end

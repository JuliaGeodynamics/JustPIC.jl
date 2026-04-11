@static if ENV["JULIA_JUSTPIC_BACKEND"] === "AMDGPU"
    using AMDGPU
    AMDGPU.allowscalar(true)
elseif ENV["JULIA_JUSTPIC_BACKEND"] === "CUDA"
    using CUDA
    CUDA.allowscalar(true)
end

using ParallelStencil

@static if ENV["JULIA_JUSTPIC_BACKEND"] === "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 2)
elseif ENV["JULIA_JUSTPIC_BACKEND"] === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

using JustPIC, JustPIC._2D, CellArrays, Test, LinearAlgebra

const backend = @static if ENV["JULIA_JUSTPIC_BACKEND"] === "AMDGPU"
    JustPIC.AMDGPUBackend
elseif ENV["JULIA_JUSTPIC_BACKEND"] === "CUDA"
    CUDABackend
else
    JustPIC.CPUBackend
end

function expand_range(x::LinRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = x1 - dx
    xF = x2 + dx
    return LinRange(xI, xF, n + 2)
end

function expand_range(x::AbstractVector)
    dx_left = x[2] - x[1]
    dx_right = x[end] - x[end - 1]
    x1, x2 = extrema(x)
    xI = x1 - dx_left
    xF = x2 + dx_right
    return vcat(xI, x, xF)
end

# Analytical flow solution
vx_stream(x, y) = 250 * sin(π * x) * cos(π * y)
vy_stream(x, y) = -250 * cos(π * x) * sin(π * y)

# Analytical flow solution
vi_stream(x) = π * 1.0e-5 * (x - 0.5)

@testset "Subgrid diffusion 2D" begin
    nxcell, max_xcell, min_xcell = 12, 12, 1
    n = 5 # number of vertices
    nx = ny = n - 1
    Lx = Ly = 1.0
    # nodal vertices
    xvi = xv, yv = LinRange(0, Lx, n), LinRange(0, Ly, n)
    dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
    # nodal centers
    xc, yc = LinRange(0 + dx / 2, Lx - dx / 2, n - 1), LinRange(0 + dy / 2, Ly - dy / 2, n - 1)
    # staggered grid velocity nodal locations
    grid_vx = xv, expand_range(yc)
    grid_vy = expand_range(xc), yv
    grid_vel = grid_vx, grid_vx
    # Initialize particles & particle fields
    particles = _2D.init_particles(
        backend, nxcell, max_xcell, min_xcell, grid_vel...,
    )

    arrays = _2D.SubgridDiffusionCellArrays(particles)
    # Test they are allocated in the right backend
    @test arrays.ΔT_subgrid isa TA(backend)
    @test arrays.pT0.data isa TA(backend)
    @test arrays.pΔT.data isa TA(backend)
    @test arrays.dt₀.data isa TA(backend)

    @test_throws ArgumentError SubgridDiffusionCellArrays(1)

end

@testset "Particles initialization 2D" begin
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 12, 24, 6
    n = 101
    nx = ny = n - 1
    ni = nx, ny
    Lx = Ly = 1.0
    # nodal vertices
    xvi = xv, yv = LinRange(0, Lx, n), LinRange(0, Ly, n)
    dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
    # nodal centers
    xc, yc = LinRange(0 + dx / 2, Lx - dx / 2, n - 1), LinRange(0 + dy / 2, Ly - dy / 2, n - 1)
    # staggered grid velocity nodal locations
    grid_vx = xv, expand_range(yc)
    grid_vy = expand_range(xc), yv

    particles1 = _2D.init_particles(
        backend, nxcell, max_xcell, min_xcell, (grid_vx, grid_vy)...,
    )

    particles2 = _2D.init_particles(
        backend, nxcell, max_xcell, min_xcell, (grid_vx, grid_vy)...,
    )

    @test particles1.min_xcell == particles2.min_xcell
    @test particles1.max_xcell == particles2.max_xcell
    @test particles1.np == particles2.np
end

@testset "Cell index 2D" begin
    n = 11
    x = range(0, stop = 1, length = n)
    xv = x, x

    px = rand()
    idx = _2D.cell_index(px, x)
    @test x[idx] ≤ px < x[idx + 1]

    px, py = rand(2)
    i, j = _2D.cell_index((px, py), xv)
    @test x[i] ≤ px < x[i + 1]
    @test x[j] ≤ py < x[j + 1]

    x = range(0, stop = 1, length = n)
    y = range(-1, stop = 0, length = n)
    px, py = rand(), -rand()
    idx = cell_index(py, y)
    @test y[idx] ≤ py < y[idx + 1]

    xv = x, y
    i, j = _2D.cell_index((px, py), xv)
    @test x[i] ≤ px < x[i + 1]
    @test y[j] ≤ py < y[j + 1]
end

@testset "Refined grid advection helpers 2D" begin
    xv = TA(backend)([0.0, 0.1, 0.3, 0.6, 1.0])
    yv = TA(backend)(collect(LinRange(0.0, 1.0, 5)))
    xc = TA(backend)([(xv[i] + xv[i + 1]) / 2 for i in 1:(length(xv) - 1)])
    yc = TA(backend)([(yv[i] + yv[i + 1]) / 2 for i in 1:(length(yv) - 1)])

    grid_vx = xv, TA(backend)(expand_range(Array(yc)))
    grid_vy = TA(backend)(expand_range(Array(xc))), yv
    grid_vi = grid_vx, grid_vy
    dxi_velocity = JustPIC._2D.compute_dx.(grid_vi)
    dxi_vertex = JustPIC._2D.compute_dx((xv, yv))

    p = (0.22, 0.48)
    idx = (3, 3)
    corrected_idx = (
        JustPIC._2D.find_parent_cell_bisection(p[1], xv, idx[1]),
        JustPIC._2D.find_parent_cell_bisection(p[2], yv, idx[2]),
    )

    Vx = TA(backend)([2.0 * x + y for x in Array(grid_vx[1]), y in Array(grid_vx[2])])
    Vy = TA(backend)([x - 3.0 * y for x in Array(grid_vy[1]), y in Array(grid_vy[2])])

    v_linp = JustPIC._2D.interp_velocity2particle_LinP(p, grid_vi, dxi_velocity, (Vx, Vy), idx)
    v_mqs = JustPIC._2D.interp_velocity2particle_MQS(p, grid_vi, dxi_velocity, (Vx, Vy), idx)
    v_linp_corrected = JustPIC._2D.interp_velocity2particle_LinP(p, grid_vi, dxi_velocity, (Vx, Vy), corrected_idx)
    v_mqs_corrected = JustPIC._2D.interp_velocity2particle_MQS(p, grid_vi, dxi_velocity, (Vx, Vy), corrected_idx)

    @test all(v_linp .≈ v_linp_corrected)
    @test all(v_mqs .≈ v_mqs_corrected)

    F0 = TA(backend)([x + 2.0 * y for x in Array(xv), y in Array(yv)])
    F_linp = similar(F0)
    F_mqs = similar(F0)
    copyto!(F_linp, F0)
    copyto!(F_mqs, F0)

    Vx_const = TA(backend)(fill(0.08, size(Vx)))
    Vy_const = TA(backend)(fill(0.02, size(Vy)))
    dt = 1.0

    semilagrangian_advection_LinP!((F_linp,), (F0,), RungeKutta2(), (Vx_const, Vy_const), grid_vi, (xv, yv), dt)
    semilagrangian_advection_MQS!((F_mqs,), (F0,), RungeKutta2(), (Vx_const, Vy_const), grid_vi, (xv, yv), dt)

    p_backtrack = (xv[3] - 0.08, yv[3] - 0.02)
    I_backtrack = (
        JustPIC._2D.find_parent_cell_bisection(p_backtrack[1], xv, 3),
        JustPIC._2D.find_parent_cell_bisection(p_backtrack[2], yv, 3),
    )
    di_backtrack = (dxi_vertex[1][I_backtrack[1]], dxi_vertex[2][I_backtrack[2]])
    expected = JustPIC._2D._grid2particle(
        p_backtrack, (xv, yv), di_backtrack, F0, I_backtrack
    )
    @test F_linp[3, 3] ≈ expected atol = 1.0e-12 rtol = 1.0e-12
    @test F_mqs[3, 3] ≈ expected atol = 1.0e-12 rtol = 1.0e-12
    @test F_linp[3, 3] ≈ F_mqs[3, 3] atol = 1.0e-12 rtol = 1.0e-12
    @test F_linp[3, 3] != F0[3, 3]
end

@testset "Refined grid particle initialization 2D" begin
    xv = [0.0, 0.1, 0.3, 0.6, 1.0]
    yv = [0.0, 0.2, 0.5, 0.7, 1.0]
    xvi = (xv, yv)
    xci = (
        [(xv[i] + xv[i + 1]) / 2 for i in 1:(length(xv) - 1)],
        [(yv[i] + yv[i + 1]) / 2 for i in 1:(length(yv) - 1)],
    )
    grid_vi = (
        (xv, expand_range(xci[2])),
        (expand_range(xci[1]), yv),
    )

    nxcell, max_xcell, min_xcell = 8, 12, 4
    particles = _2D.init_particles(
        backend, nxcell, max_xcell, min_xcell, grid_vi...,
    )

    @test Array.(particles.xvi) == xvi
    @test Array.(particles.xci) == xci
    @test Array.(particles.xi_vel[1]) == grid_vi[1]
    @test Array.(particles.xi_vel[2]) == grid_vi[2]
end

@testset "Passive markers 2D" begin
    # Initialize particles -------------------------------
    n = 51
    Lx = Ly = 1.0
    # nodal vertices
    xvi = xv, yv = LinRange(0, Lx, n), LinRange(0, Ly, n)
    dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
    # nodal centers
    xc, yc = LinRange(0 + dx / 2, Lx - dx / 2, n - 1), LinRange(0 + dy / 2, Ly - dy / 2, n - 1)
    # staggered grid velocity nodal locations
    grid_vx = xv, expand_range(yc)
    grid_vy = expand_range(xc), yv

    # Cell fields -------------------------------
    Vx = TA(backend)([-vi_stream(y) for x in grid_vx[1], y in grid_vx[2]])
    Vy = TA(backend)([ vi_stream(x) for x in grid_vy[1], y in grid_vy[2]])

    T = TA(backend)([y for x in xv, y in yv])
    P = TA(backend)([x for x in xv, y in yv])
    V = Vx, Vy

    w = π * 1.0e-5  # angular velocity
    period = 1  # revolution number
    tmax = period / (w / (2 * π))
    dt = 200.0

    np = 256 # number of passive markers
    passive_coords = ntuple(Val(2)) do i
        TA(backend)((rand(np) .+ 1) .* Lx / 4)
    end

    passive_markers = init_passive_markers(backend, passive_coords)
    T_marker = TA(backend)(zeros(np))
    P_marker = TA(backend)(zeros(np))

    for _ in 1:50
        _2D.advection!(passive_markers, RungeKutta2(2 / 3), V, (grid_vx, grid_vy), dt)
    end

    # interpolate grid fields T and P onto the marker locations
    _2D.grid2particle!((T_marker, P_marker), xvi, (T, P), passive_markers)
    x_marker = passive_markers.coords[1]
    y_marker = passive_markers.coords[2]

    @test y_marker ≈ T_marker
    @test x_marker ≈ P_marker
end

@testset "Pure shear 2D" begin

    @parallel_indices (I...) function InitialFieldsParticles!(phases, px, py, index)
        for ip in cellaxes(phases)
            # quick escape
            @index(index[ip, I...]) == 0 && continue
            x = @index px[ip, I...]
            y = @index py[ip, I...]
            if x < y
                @index phases[ip, I...] = 1.0
            else
                @index phases[ip, I...] = 2.0
            end
        end
        return nothing
    end

    year = 365 * 3600 * 24
    L = (x = 1.0, y = 1.0)
    Nc = (x = 128, y = 128)
    Nv = (x = Nc.x + 1, y = Nc.y + 1)
    Δ = (x = L.x / Nc.x, y = L.y / Nc.y)
    Nt = 20
    Nout = 1
    C = 0.25

    verts = (x = LinRange(-L.x / 2, L.x / 2, Nv.x), y = LinRange(-L.y / 2, L.y / 2, Nv.y))
    cents = (x = LinRange(-L.x / 2 + Δ.x / 2, L.x / 2 - Δ.x / 2, Nc.x), y = LinRange(-L.y / 2 + Δ.y / 2, L.y / 2 - Δ.y / 2, Nc.y))
    cents_ext = (x = LinRange(-L.x / 2 - Δ.x / 2, L.x / 2 + Δ.x / 2, Nc.x + 2), y = LinRange(-L.y / 2 - Δ.y / 2, L.y / 2 + Δ.y / 2, Nc.y + 2))
    size_x = (Nc.x + 1, Nc.y + 2)
    size_y = (Nc.x + 2, Nc.y + 1)
    V = (
        x = @zeros(size_x),
        y = @zeros(size_y),
    )

    # Set velocity field
    ε̇bg = -1.0
    for i in 1:size(V.x, 1),  j in 1:size(V.x, 2)
        V.x[i, j] = verts.x[i] * ε̇bg
    end

    for i in 1:size(V.y, 1),  j in 1:size(V.y, 2)
        V.y[i, j] = -verts.y[j] * ε̇bg
    end

    grid_vx = (verts.x, cents_ext.y)
    grid_vy = (cents_ext.x, verts.y)

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 60, 80, 50
    particles = init_particles(
        backend,
        nxcell,
        max_xcell,
        min_xcell,
        (grid_vx, grid_vy)...,
    ) # random position by default

    # Initialise phase field
    particle_args = phases, = init_cell_arrays(particles, Val(1))  # cool

    @parallel InitialFieldsParticles!(phases, particles.coords..., particles.index)

    phase_ratios = JustPIC._2D.PhaseRatios(backend, 2, values(Nc))
    update_phase_ratios!(phase_ratios, particles, phases)

    @test all(extrema(sum(phase_ratios.vertex.data, dims = 2)) .≈ 1)
    @test all(extrema(sum(phase_ratios.center.data, dims = 2)) .≈ 1)
    @test all(extrema(sum(phase_ratios.Vx.data, dims = 2)) .≈ 1)
    @test all(extrema(sum(phase_ratios.Vy.data, dims = 2)) .≈ 1)

    # Time step
    t = 0.0e0
    Δt = C * min(Δ...) / max(maximum(abs.(V.x)), maximum(abs.(V.y)))

    # Create necessary tuples

    Vxc = 0.5 * (V.x[1:(end - 1), 2:(end - 1)] .+ V.x[2:(end - 0), 2:(end - 1)])
    Vyc = 0.5 * (V.y[2:(end - 1), 1:(end - 1)] .+ V.y[2:(end - 1), 2:(end - 0)])

    for it in 1:Nt
        @show it
        advection!(particles, RungeKutta2(), values(V), Δt)
        move_particles!(particles, particle_args)
        inject_particles_phase!(particles, phases, (), ())
        update_phase_ratios!(phase_ratios, particles, phases)
    end

    @test all(extrema(sum(phase_ratios.vertex.data, dims = 2)) .≈ 1)
    @test all(extrema(sum(phase_ratios.center.data, dims = 2)) .≈ 1)
    @test all(extrema(sum(phase_ratios.Vx.data, dims = 2)) .≈ 1)
    @test all(extrema(sum(phase_ratios.Vy.data, dims = 2)) .≈ 1)
end

function advection_test_2D()
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 25, 50, 10
    n = 65
    nx = ny = n - 1
    Lx = Ly = 1.0
    # nodal vertices
    xvi = xv, yv = LinRange(0, Lx, n), LinRange(0, Ly, n)
    dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
    # nodal centers
    xci = xc, yc = LinRange(0 + dx / 2, Lx - dx / 2, n - 1), LinRange(0 + dy / 2, Ly - dy / 2, n - 1)
    # staggered grid velocity nodal locations
    grid_vx = xv, expand_range(yc)
    grid_vy = expand_range(xc), yv

    particles = _2D.init_particles(
        backend, nxcell, max_xcell, min_xcell, (grid_vx, grid_vy)...,
    )

    # Cell fields -------------------------------
    Vx = TA(backend)([vx_stream(x, y) for x in grid_vx[1], y in grid_vx[2]])
    Vy = TA(backend)([vy_stream(x, y) for x in grid_vy[1], y in grid_vy[2]])
    T = TA(backend)([y for x in xv, y in yv])
    T0 = deepcopy(T)
    V = Vx, Vy

    dt = min(dx / maximum(abs.(Array(Vx))), dy / maximum(abs.(Array(Vy)))) / 2

    # Advection test
    particle_args = pT, = init_cell_arrays(particles, Val(1))
    _2D.grid2particle!(pT, T, particles)

    sumT = sum(T)

    niter = 25
    for it in 1:niter
        _2D.particle2grid!(T, pT, particles)
        copyto!(T0, T)
        _2D.advection!(particles, RungeKutta2(2 / 3), V, dt)
        _2D.move_particles!(particles, particle_args)
        _2D.inject_particles!(particles, (pT,))
        _2D.grid2particle!(pT, T, particles)
    end

    sumT_final = sum(T)

    return abs(sumT - sumT_final) / sumT

end

function test_advection_2D()
    err = advection_test_2D()
    tol = 1.0e-2
    passed = err < tol

    return passed
end

function advection_test_2D_refined()
    nxcell, max_xcell, min_xcell = 25, 50, 10
    xv = [0.0, 0.05, 0.12, 0.21, 0.32, 0.45, 0.6, 0.77, 0.91, 1.0]
    yv = [0.0, 0.04, 0.1, 0.18, 0.29, 0.43, 0.58, 0.74, 0.89, 1.0]
    xvi = (xv, yv)
    xc = [(xv[i] + xv[i + 1]) / 2 for i in 1:(length(xv) - 1)]
    yc = [(yv[i] + yv[i + 1]) / 2 for i in 1:(length(yv) - 1)]
    grid_vx = xv, expand_range(yc)
    grid_vy = expand_range(xc), yv

    particles = _2D.init_particles(
        backend, nxcell, max_xcell, min_xcell, grid_vx, grid_vy,
    )

    Vx = TA(backend)([vx_stream(x, y) for x in grid_vx[1], y in grid_vx[2]])
    Vy = TA(backend)([vy_stream(x, y) for x in grid_vy[1], y in grid_vy[2]])
    T = TA(backend)([y for x in xv, y in yv])
    T0 = deepcopy(T)
    V = Vx, Vy

    dx_min = minimum(diff(xv))
    dy_min = minimum(diff(yv))
    dt = min(dx_min / maximum(abs.(Array(Vx))), dy_min / maximum(abs.(Array(Vy)))) / 2

    particle_args = pT, = init_cell_arrays(particles, Val(1))
    _2D.grid2particle!(pT, T, particles)

    sumT = sum(T)

    niter = 25
    for _ in 1:niter
        _2D.particle2grid!(T, pT, particles)
        copyto!(T0, T)
        _2D.advection!(particles, RungeKutta2(2 / 3), V, dt)
        _2D.move_particles!(particles, particle_args)
        _2D.inject_particles!(particles, (pT,))
        _2D.grid2particle!(pT, T, particles)
    end

    sumT_final = sum(T)

    return abs(sumT - sumT_final) / sumT
end

function test_advection_2D_refined()
    err = advection_test_2D_refined()
    tol = 1.0e-1
    passed = err < tol

    return passed
end

function test_rotating_circle()
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 25, 50, 10
    n = 256
    nx = ny = n - 1
    Lx = Ly = 1.0
    # nodal vertices
    xvi = xv, yv = LinRange(0, Lx, n), LinRange(0, Ly, n)
    dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
    # nodal centers
    xc, yc = LinRange(0 + dx / 2, Lx - dx / 2, n - 1), LinRange(0 + dy / 2, Ly - dy / 2, n - 1)
    # staggered grid velocity nodal locations
    grid_vx = xv, expand_range(yc)
    grid_vy = expand_range(xc), yv

    particles = _2D.init_particles(
        backend, nxcell, max_xcell, min_xcell, (grid_vx, grid_vy)...,
    )

    # Cell fields -------------------------------
    Vx = TA(backend)([-vi_stream(y) for x in grid_vx[1], y in grid_vx[2]])
    Vy = TA(backend)([ vi_stream(x) for x in grid_vy[1], y in grid_vy[2]])
    xc0 = yc0 = 0.25
    R = 6 * dx
    T = TA(backend)([((x - xc0)^2 + (y - yc0)^2 ≤ R^2) * 1.0 for x in xv, y in yv])
    T0 = deepcopy(T)
    V = Vx, Vy

    w = π * 1.0e-5  # angular velocity
    period = 1  # revolution number
    tmax = period / (w / (2 * π)) / 10
    dt = 200.0

    particle_args = pT, = init_cell_arrays(particles, Val(1))
    _2D.grid2particle!(pT, T, particles)

    t = 0
    it = 0
    sumT = sum(T)
    while t ≤ tmax
        _2D.particle2grid!(T, pT, particles)
        copyto!(T0, T)
        _2D.advection!(particles, _2D.RungeKutta2(), V, dt)
        _2D.move_particles!(particles, particle_args)
        _2D.inject_particles!(particles, (pT,))
        _2D.grid2particle!(pT, T, particles)
        t += dt
        it += 1
    end

    sumT_final = sum(T)

    return abs(sumT - sumT_final) / sumT
end

function test_rotation_2D()
    err = test_rotating_circle()
    tol = 1.0e-1
    passed = err < tol

    return passed
end

@testset "Miniapps" begin
    @testset "1. Advection 2D" begin
        @test test_advection_2D()
    end

    @testset "1b. Advection 2D refined grid" begin
        @test test_advection_2D_refined()
    end

    @testset "2. Rotating circle 2D" begin
        @test test_rotation_2D()
    end
end

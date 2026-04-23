@static if ENV["JULIA_JUSTPIC_BACKEND"] === "AMDGPU"
    using AMDGPU
    AMDGPU.allowscalar(true)
elseif ENV["JULIA_JUSTPIC_BACKEND"] === "CUDA"
    using CUDA
    CUDA.allowscalar(true)
end

using JustPIC, JustPIC._3D, CellArrays, ParallelStencil, Test, LinearAlgebra

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
vx_stream(x, z) = 250 * sin(π * x) * cos(π * z)
vy_stream(x, z) = 0.0
vz_stream(x, z) = -250 * cos(π * x) * sin(π * z)

struct ForceInjectionPoint3D{T}
    coords::NTuple{3, T}
    active::Bool
end

Base.isnan(p::ForceInjectionPoint3D) = !p.active
Base.getindex(p::ForceInjectionPoint3D, i::Int) = p.coords[i]
@testset "Interpolations 3D" begin
    nxcell, max_xcell, min_xcell = 16, 16, 1
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
    grid_vx = xv, expand_range(yc), expand_range(zc)
    grid_vy = expand_range(xc), yv, expand_range(zc)
    grid_vz = expand_range(xc), expand_range(yc), zv
    grid_vel = grid_vx, grid_vy, grid_vz
    # Initialize particles -------------------------------
    particles = _3D.init_particles(
        backend, nxcell, max_xcell, min_xcell, grid_vel...
    )
    pT, = _3D.init_cell_arrays(particles, Val(1))
    xvi_p = JustPIC._3D.add_periodic_ghost_nodes.(xvi)
    # Linear field at the vertices
    T = TA(backend)([z for x in xvi_p[1], y in xvi_p[2], z in xvi_p[3]])
    T0 = TA(backend)([z for x in xvi_p[1], y in xvi_p[2], z in xvi_p[3]])
    # Grid to particle test
    _3D.grid2particle!(pT, xvi_p, T, particles, diff.(xvi_p))
    active = Array(particles.index.data)
    @test Array(pT.data)[active] ≈ Array(particles.coords[3].data)[active]
    # Grid to particle test
    _3D.grid2particle_flip!(pT, xvi_p, T, T0, particles)
    @test Array(pT.data)[active] ≈ Array(particles.coords[3].data)[active]
    # Particle to grid test
    T2 = similar(T)
    _3D.particle2grid!(T2, pT, particles)
    finite_mask = isfinite.(T2)
    @test norm(T2[finite_mask] .- T[finite_mask]) / count(finite_mask) < 1.0e-1
    # test copy function
    particles_copy = copy(particles)
    pT_copy = copy(pT)
    @test particles_copy.index.data[:] == particles.index.data[:]
    @test pT_copy.data[:] == pT.data[:]
    GC.gc()
end

@testset "Particles initialization 3D" begin
    nxcell, max_xcell, min_xcell = 24, 24, 1
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
    grid_vx = xv, expand_range(yc), expand_range(zc)
    grid_vy = expand_range(xc), yv, expand_range(zc)
    grid_vz = expand_range(xc), expand_range(yc), zv
    grid_vel = grid_vx, grid_vy, grid_vz
    # Initialize particles -------------------------------
    particles1 = _3D.init_particles(
        backend, nxcell, max_xcell, min_xcell, grid_vel...
    )
    particles2 = _3D.init_particles(
        backend, nxcell, max_xcell, min_xcell, grid_vel...
    )
    @test particles1.min_xcell == particles2.min_xcell
    @test particles1.max_xcell == particles2.max_xcell
    @test particles1.np == particles2.np
    GC.gc()
end

@testset "Subgrid diffusion 3D" begin
    nxcell, max_xcell, min_xcell = 24, 24, 1
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
    grid_vx = xv, expand_range(yc), expand_range(zc)
    grid_vy = expand_range(xc), yv, expand_range(zc)
    grid_vz = expand_range(xc), expand_range(yc), zv
    grid_vel = grid_vx, grid_vy, grid_vz
    # Initialize particles -------------------------------
    particles = _3D.init_particles(
        backend, nxcell, max_xcell, min_xcell, grid_vel...
    )
    arrays = _3D.SubgridDiffusionCellArrays(particles)
    # Test they are allocated in the right backend
    @test arrays.ΔT_subgrid isa TA(backend)
    @test arrays.pT0.data isa TA(backend)
    @test arrays.pΔT.data isa TA(backend)
    @test arrays.dt₀.data isa TA(backend)
    GC.gc()
end

@testset "Cell index 3D" begin
    n = 100
    a, b = rand() * 50, rand() * 50
    start, finish = extrema((a, b))
    L = finish - start
    x = range(start, stop = finish, length = n)
    xv = x, x, x
    p = px, py, pz = tuple((rand(3) .* L .+ start)...)
    i, j, k = _3D.cell_index(p, xv)
    @test x[i] ≤ px < x[i + 1]
    @test x[j] ≤ py < x[j + 1]
    @test x[k] ≤ pz < x[k + 1]
    y = x
    z = range(-start, stop = finish, length = n)
    xv = x, y, z
    px, py = tuple((rand(2) .* L .+ start)...)
    Lz = z[end] - z[1]
    pz = rand() * Lz - start
    p = px, py, pz
    i, j, k = _3D.cell_index(p, xv)
    @test x[i] ≤ px < x[i + 1]
    @test y[j] ≤ py < y[j + 1]
    @test z[k] ≤ pz < z[k + 1]
    GC.gc()
end

@testset "Periodic ghost nodes 3D" begin
    zv_uniform = LinRange(-1.0, 1.0, 5)
    zv_uniform_periodic = JustPIC._3D.add_periodic_ghost_nodes(zv_uniform)
    @test zv_uniform_periodic isa LinRange
    @test Array(zv_uniform_periodic) ≈ [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]

    zv_refined = [-1.0, -0.7, -0.2, 0.4, 1.0]
    zv_refined_periodic = JustPIC._3D.add_periodic_ghost_nodes(zv_refined)
    @test zv_refined_periodic isa Vector
    @test zv_refined_periodic == [-1.6, -1.0, -0.7, -0.2, 0.4, 1.0, 1.3]
end

@testset "Refined grid particle initialization 3D" begin
    xv = [0.0, 0.1, 0.25, 0.55, 1.0]
    yv = [0.0, 0.2, 0.45, 0.8, 1.0]
    zv = [0.0, 0.15, 0.35, 0.65, 1.0]
    xvi = (xv, yv, zv)
    xci = ntuple(i -> [(xvi[i][j] + xvi[i][j + 1]) / 2 for j in 1:(length(xvi[i]) - 1)], Val(3))
    grid_vi = (
        (xv, expand_range(xci[2]), expand_range(xci[3])),
        (expand_range(xci[1]), yv, expand_range(xci[3])),
        (expand_range(xci[1]), expand_range(xci[2]), zv),
    )

    nxcell, max_xcell, min_xcell = 8, 12, 4
    particles = _3D.init_particles(
        backend, nxcell, max_xcell, min_xcell, grid_vi...,
    )

    @test Array.(particles.xvi) == JustPIC._3D.add_periodic_ghost_nodes.(xvi)
    @test Array.(particles.xci) == JustPIC._3D.add_periodic_ghost_nodes.(xci)
    @test Array.(particles.xi_vel[1]) == grid_vi[1]
    @test Array.(particles.xi_vel[2]) == grid_vi[2]
    @test Array.(particles.xi_vel[3]) == grid_vi[3]
end

@testset "Passive markers 3D" begin
    n = 32
    nx = ny = nz = n - 1
    Lx = Ly = Lz = 1.0
    ni = nx, ny, nz
    Li = Lx, Ly, Lz
    # nodal vertices
    xvi = xv, yv, zv = ntuple(i -> LinRange(0, Li[i], n), Val(3))
    # grid spacing
    dxi = dx, dy, dz = ntuple(i -> xvi[i][2] - xvi[i][1], Val(3))
    # nodal centers
    xci = xc, yc, zc = ntuple(i -> LinRange(0 + dxi[i] / 2, Li[i] - dxi[i] / 2, ni[i]), Val(3))

    # staggered grid velocity nodal locations
    grid_vx = xv, expand_range(yc), expand_range(zc)
    grid_vy = expand_range(xc), yv, expand_range(zc)
    grid_vz = expand_range(xc), expand_range(yc), zv
    grid_vel = grid_vx, grid_vy, grid_vz

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 24, 24, 3
    particles = _3D.init_particles(
        backend, nxcell, max_xcell, min_xcell, grid_vel...
    )

    # Cell fields -------------------------------
    Vx = TA(backend)([vx_stream(x, z) for x in grid_vx[1], y in grid_vx[2], z in grid_vx[3]])
    Vy = TA(backend)([vy_stream(x, z) for x in grid_vy[1], y in grid_vy[2], z in grid_vy[3]])
    Vz = TA(backend)([vz_stream(x, z) for x in grid_vz[1], y in grid_vz[2], z in grid_vz[3]])
    T = TA(backend)([z for x in xv, y in yv, z in zv])
    P = TA(backend)([x for x in xv, y in yv, z in zv])
    V = Vx, Vy, Vz

    dt = min(dx / maximum(abs.(Vx)), dy / maximum(abs.(Vy)), dz / maximum(abs.(Vz))) / 2
    np = 256 # number of passive markers

    passive_coords = ntuple(Val(3)) do i
        TA(backend)((rand(np) .+ 1) .* Lx / 4)
    end

    passive_markers = _3D.init_passive_markers(backend, passive_coords)
    T_marker = TA(backend)(zeros(np))
    P_marker = TA(backend)(zeros(np))

    for _ in 1:75
        _3D.advection!(passive_markers, _3D.RungeKutta2(2 / 3), V, (grid_vx, grid_vy, grid_vz), dt)
    end

    # interpolate grid fields T and P onto the marker locations
    _3D.grid2particle!((T_marker, P_marker), xvi, (T, P), passive_markers)
    x_marker = passive_markers.coords[1]
    z_marker = passive_markers.coords[3]

    @test x_marker ≈ P_marker
    @test z_marker ≈ T_marker
    GC.gc()
end


@testset "Forced injection 3D" begin
    nxcell, max_xcell, min_xcell = 0, 4, 0

    n = 3
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
    grid_vx = xv, expand_range(yc), expand_range(zc)
    grid_vy = expand_range(xc), yv, expand_range(zc)
    grid_vz = expand_range(xc), expand_range(yc), zv
    grid_vel = grid_vx, grid_vy, grid_vz

    particles = _3D.init_particles(backend, nxcell, max_xcell, min_xcell, grid_vel...)
    pphase, = _3D.init_cell_arrays(particles, Val(1))

    ni = size(particles.index)
    nslots = _3D.cellnum(particles.index)
    p_invalid = ForceInjectionPoint3D((0.0, 0.0, 0.0), false)
    p_new = TA(backend)(fill(p_invalid, ni..., nslots))
    for i in 1:ni[1], j in 1:ni[2], k in 1:ni[3], c in 1:nslots
        p_new[i, j, k, c] = ForceInjectionPoint3D((0.1 * c + 0.01 * i, 0.1 * c + 0.01 * j, 0.1 * c + 0.01 * k), true)
    end

    _3D.force_injection!(particles, p_new, (pphase,), (5.0,))

    x_data = vec(Array(particles.coords[1].data))
    y_data = vec(Array(particles.coords[2].data))
    z_data = vec(Array(particles.coords[3].data))
    x_expected = vec([p_new[i, j, k, c][1] for i in 1:ni[1], j in 1:ni[2], k in 1:ni[3], c in 1:nslots])
    y_expected = vec([p_new[i, j, k, c][2] for i in 1:ni[1], j in 1:ni[2], k in 1:ni[3], c in 1:nslots])
    z_expected = vec([p_new[i, j, k, c][3] for i in 1:ni[1], j in 1:ni[2], k in 1:ni[3], c in 1:nslots])

    @test all(Array(particles.index.data))
    @test all(Array(pphase.data) .== 5.0)
    @test sort(x_data) ≈ sort(x_expected)
    @test sort(y_data) ≈ sort(y_expected)
    @test sort(z_data) ≈ sort(z_expected)

    particles_no_fields = _3D.init_particles(backend, nxcell, max_xcell, min_xcell, grid_vel...)
    _3D.force_injection!(particles_no_fields, p_new)
    @test all(Array(particles_no_fields.index.data))

    particles_skip = _3D.init_particles(backend, nxcell, max_xcell, min_xcell, grid_vel...)
    p_empty = TA(backend)(fill(p_invalid, ni..., nslots))
    _3D.force_injection!(particles_skip, p_empty)
    @test !any(Array(particles_skip.index.data))
end

function test_advection_3D()

    n = 64
    nx = ny = nz = n - 1
    Lx = Ly = Lz = 1.0
    ni = nx, ny, nz
    Li = Lx, Ly, Lz
    # nodal vertices
    xvi = xv, yv, zv = ntuple(i -> LinRange(0, Li[i], n), Val(3))
    # grid spacing
    dxi = dx, dy, dz = ntuple(i -> xvi[i][2] - xvi[i][1], Val(3))
    # nodal centers
    xci = xc, yc, zc = ntuple(i -> LinRange(0 + dxi[i] / 2, Li[i] - dxi[i] / 2, ni[i]), Val(3))

    # staggered grid velocity nodal locations
    grid_vx = xv, expand_range(yc), expand_range(zc)
    grid_vy = expand_range(xc), yv, expand_range(zc)
    grid_vz = expand_range(xc), expand_range(yc), zv
    grid_vel = grid_vx, grid_vy, grid_vz

    # Cell fields -------------------------------
    Vx = TA(backend)([vx_stream(x, z) for x in grid_vx[1], y in grid_vx[2], z in grid_vx[3]])
    Vy = TA(backend)([vy_stream(x, z) for x in grid_vy[1], y in grid_vy[2], z in grid_vy[3]])
    Vz = TA(backend)([vz_stream(x, z) for x in grid_vz[1], y in grid_vz[2], z in grid_vz[3]])
    xvi_p = JustPIC._3D.add_periodic_ghost_nodes.(xvi)
    T = TA(backend)([z for x in xvi_p[1], y in xvi_p[2], z in xvi_p[3]])
    T0 = deepcopy(T)
    V = Vx, Vy, Vz
    dt = min(
        dx / maximum(abs.(Vx)),
        dy / maximum(abs.(Vy)),
        dz / maximum(abs.(Vz))
    ) / 4

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 125, 150, 100
    particles = _3D.init_particles(
        backend, nxcell, max_xcell, min_xcell, grid_vel...
    )

    # Advection test
    particle_args = pT, = _3D.init_cell_arrays(particles, Val(1))
    _3D.grid2particle!(pT, xvi_p, T, particles, diff.(xvi_p))
    sumT = sum(T)

    niter = 5
    for _ in 1:niter
        _3D.particle2grid!(T, pT, particles)
        copyto!(T0, T)
        _3D.advection!(particles, _3D.RungeKutta2(), V, dt)
        _3D.move_particles!(particles, particle_args)
        # reseed
        _3D.inject_particles!(particles, (pT,))
        _3D.grid2particle!(pT, xvi_p, T, particles, diff.(xvi_p))
    end
    sumT_final = sum(T)
    err = abs(sumT - sumT_final) / sumT
    println(err)
    return err
end

function test_advection_3D_refined()
    xv = [0.0, 0.04, 0.09, 0.16, 0.25, 0.37, 0.52, 0.68, 0.84, 1.0]
    yv = [0.0, 0.05, 0.11, 0.2, 0.31, 0.44, 0.58, 0.73, 0.87, 1.0]
    zv = [0.0, 0.03, 0.08, 0.15, 0.26, 0.4, 0.56, 0.72, 0.88, 1.0]
    xvi = (xv, yv, zv)
    xc = [(xv[i] + xv[i + 1]) / 2 for i in 1:(length(xv) - 1)]
    yc = [(yv[i] + yv[i + 1]) / 2 for i in 1:(length(yv) - 1)]
    zc = [(zv[i] + zv[i + 1]) / 2 for i in 1:(length(zv) - 1)]

    grid_vx = xv, expand_range(yc), expand_range(zc)
    grid_vy = expand_range(xc), yv, expand_range(zc)
    grid_vz = expand_range(xc), expand_range(yc), zv

    Vx = TA(backend)([vx_stream(x, z) for x in grid_vx[1], y in grid_vx[2], z in grid_vx[3]])
    Vy = TA(backend)([vy_stream(x, z) for x in grid_vy[1], y in grid_vy[2], z in grid_vy[3]])
    Vz = TA(backend)([vz_stream(x, z) for x in grid_vz[1], y in grid_vz[2], z in grid_vz[3]])
    xvi_p = JustPIC._3D.add_periodic_ghost_nodes.(xvi)
    T = TA(backend)([z for x in xvi_p[1], y in xvi_p[2], z in xvi_p[3]])
    T0 = deepcopy(T)
    V = Vx, Vy, Vz

    dx_min = minimum(diff(xv))
    dy_min = minimum(diff(yv))
    dz_min = minimum(diff(zv))
    dt = min(
        dx_min / maximum(abs.(Vx)),
        dy_min / maximum(abs.(Vy)),
        dz_min / maximum(abs.(Vz)),
    ) / 4

    nxcell, max_xcell, min_xcell = 125, 150, 100
    particles = _3D.init_particles(
        backend, nxcell, max_xcell, min_xcell, grid_vx, grid_vy, grid_vz,
    )

    particle_args = pT, = _3D.init_cell_arrays(particles, Val(1))
    _3D.grid2particle!(pT, xvi_p, T, particles, diff.(xvi_p))
    sumT = sum(T)

    niter = 5
    for _ in 1:niter
        _3D.particle2grid!(T, pT, particles)
        copyto!(T0, T)
        _3D.advection!(particles, _3D.RungeKutta2(), V, dt)
        _3D.move_particles!(particles, particle_args)
        _3D.inject_particles!(particles, (pT,))
        _3D.grid2particle!(pT, xvi_p, T, particles, diff.(xvi_p))
    end

    sumT_final = sum(T)
    err = abs(sumT - sumT_final) / sumT
    println(err)
    return err
end

function test_advection()
    err = 0.0e0
    for _ in 1:5
        err = test_advection_3D()
        !isnan(err) && break
    end
    tol = 1.0e-1
    passed = err < tol
    return passed
end

function test_advection_refined()
    err = 0.0e0
    for _ in 1:5
        err = test_advection_3D_refined()
        !isnan(err) && break
    end
    tol = 1.0e-1
    passed = err < tol
    return passed
end

@testset "Miniapps" begin
    # @test test_advection()
    # @test test_advection_refined()
end

const BACKEND_NAME = get(ENV, "JULIA_JUSTPIC_BACKEND", "CPU")

@static if BACKEND_NAME == "AMDGPU"
    using AMDGPU
elseif BACKEND_NAME == "CUDA"
    using CUDA
elseif BACKEND_NAME == "Metal"
    using Metal
end

using JustPIC, CellArrays, Test, LinearAlgebra
import CellArraysIndexing as CAI
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
vx_stream_3D(x, z) = 250 * sin(π * x) * cos(π * z)
vy_stream_3D(x, z) = zero(x)
vz_stream_3D(x, z) = -250 * cos(π * x) * sin(π * z)

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
    Lx = Ly = Lz = FT(1)
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
    particles = JustPIC.init_particles(
        backend, nxcell, max_xcell, min_xcell, grid_vel...
    )
    pT, = JustPIC.init_cell_arrays(particles, Val(1))
    xvi_p = JustPIC.add_periodic_ghost_nodes.(xvi)
    # Linear field at the vertices
    T = TA(backend)([z for x in xvi_p[1], y in xvi_p[2], z in xvi_p[3]])
    T0 = TA(backend)([z for x in xvi_p[1], y in xvi_p[2], z in xvi_p[3]])
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
    Lx = Ly = Lz = FT(1)
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
    particles1 = JustPIC.init_particles(
        backend, nxcell, max_xcell, min_xcell, grid_vel...
    )
    particles2 = JustPIC.init_particles(
        backend, nxcell, max_xcell, min_xcell, grid_vel...
    )
    @test particles1.min_xcell == particles2.min_xcell
    @test particles1.max_xcell == particles2.max_xcell
    @test particles1.np == particles2.np
    GC.gc()
end

@testset "Regularly spaced particles initialization 3D" begin
    nxdim, max_xcell, min_xcell = (2, 3, 2), 4, 2
    n = 5 # number of vertices
    ni = ntuple(_ -> n - 1, Val(3))
    Li = ntuple(_ -> FT(1), Val(3))
    xvi = xv, yv, zv = ntuple(i -> LinRange(0, Li[i], n), Val(3))
    dxi = ntuple(i -> xvi[i][2] - xvi[i][1], Val(3))
    xc, yc, zc = ntuple(i -> LinRange(dxi[i] / 2, Li[i] - dxi[i] / 2, ni[i]), Val(3))
    grid_vx = xv, expand_range(yc), expand_range(zc)
    grid_vy = expand_range(xc), yv, expand_range(zc)
    grid_vz = expand_range(xc), expand_range(yc), zv

    particles = JustPIC.init_particles(
        backend, nxdim, max_xcell, min_xcell, grid_vx, grid_vy, grid_vz,
    )

    @test particles.nxcell == prod(nxdim)
    # `max_xcell` is raised to fit the requested layout
    @test particles.max_xcell == prod(nxdim)

    index = to_cpu(particles.index)
    coords = to_cpu.(particles.coords)
    nc = size(index)

    for I in CartesianIndices(nc)
        interior = all(i -> 1 < I[i] < nc[i], 1:3)
        @test count(index[I]) == (interior ? prod(nxdim) : 0)
    end

    # first physical cell: its lower-left corner is the first non-ghost node
    I = CartesianIndex(2, 2, 2)
    for d in 1:3
        @test sort(unique(coords[d][I])) ≈
            [xvi[d][1] + (i - FT(0.5)) * dxi[d] / nxdim[d] for i in 1:nxdim[d]]
    end
    GC.gc()
end

@testset "Particle injection skips ghost cells 3D" begin
    nxcell, max_xcell, min_xcell = 8, 12, 8
    n = 5
    nx = ny = nz = n - 1
    Lx = Ly = Lz = FT(1)
    Li = Lx, Ly, Lz
    xvi = xv, yv, zv = ntuple(i -> LinRange(0, Li[i], n), Val(3))
    dxi = dx, dy, dz = ntuple(i -> xvi[i][2] - xvi[i][1], Val(3))
    xci = xc, yc, zc = ntuple(i -> LinRange(dxi[i] / 2, Li[i] - dxi[i] / 2, (nx, ny, nz)[i]), Val(3))
    grid_vx = xv, expand_range(yc), expand_range(zc)
    grid_vy = expand_range(xc), yv, expand_range(zc)
    grid_vz = expand_range(xc), expand_range(yc), zv

    particles = JustPIC.init_particles(
        backend, nxcell, max_xcell, min_xcell, grid_vx, grid_vy, grid_vz,
    )
    pT, = JustPIC.init_cell_arrays(particles, Val(1))
    JustPIC.inject_particles!(particles, (pT,))

    index_cpu = to_cpu(particles.index)
    ghost_empty = all(
        count(index_cpu[i, j, k]) == 0 for i in axes(index_cpu, 1), j in axes(index_cpu, 2), k in axes(index_cpu, 3)
            if i in (1, size(index_cpu, 1)) || j in (1, size(index_cpu, 2)) || k in (1, size(index_cpu, 3))
    )
    @test ghost_empty
    @test count(index_cpu[2, 2, 2]) ≥ min_xcell
end

@testset "Particle injection with a cell-centered field 3D" begin
    nxcell, max_xcell, min_xcell = 12, 24, 6
    nx, ny, nz = 8, 6, 10
    ni = nx, ny, nz
    Li = FT(1), FT(1), FT(1)
    xvi = ntuple(i -> LinRange(0, Li[i], ni[i] + 1), Val(3))
    dxi = ntuple(i -> xvi[i][2] - xvi[i][1], Val(3))
    xci = ntuple(i -> LinRange(dxi[i] / 2, Li[i] - dxi[i] / 2, ni[i]), Val(3))
    grid_vx = xvi[1], expand_range(xci[2]), expand_range(xci[3])
    grid_vy = expand_range(xci[1]), xvi[2], expand_range(xci[3])
    grid_vz = expand_range(xci[1]), expand_range(xci[2]), xvi[3]

    particles = JustPIC.init_particles(
        backend, nxcell, max_xcell, min_xcell, grid_vx, grid_vy, grid_vz,
    )
    pPhases, pT = JustPIC.init_cell_arrays(particles, Val(2))
    fill!(pPhases.data, 1)

    # Empty a block of interior cells so injection fires there. It has to reach the last
    # cell along a dimension: that is where interpolating a cell-centered field reads one
    # index past its end if the field is mistaken for a vertex-centered one.
    cells = [(i, j, k) for i in 4:6, j in 3:6, k in 5:7]
    for c in cells, ip in 1:max_xcell
        CAI.@index particles.index[ip, c...] = false
    end

    # A constant field must survive interpolation exactly, so every injected particle
    # carries that constant.
    F_center = TA(backend)(fill(FT(7.5), ni))
    JustPIC.inject_particles_phase!(particles, pPhases, (pT,), (F_center,))

    index_cpu, pT_cpu = to_cpu(particles.index), to_cpu(pT)
    injected = [
        pT_cpu[c...][ip] for c in cells for ip in 1:max_xcell if index_cpu[c...][ip]
    ]
    @test !isempty(injected)
    @test all(x -> x ≈ FT(7.5), injected)
end

@testset "Subgrid diffusion 3D" begin
    nxcell, max_xcell, min_xcell = 24, 24, 1
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
    # staggered grid velocity nodal locations
    grid_vx = xv, expand_range(yc), expand_range(zc)
    grid_vy = expand_range(xc), yv, expand_range(zc)
    grid_vz = expand_range(xc), expand_range(yc), zv
    grid_vel = grid_vx, grid_vy, grid_vz
    # Initialize particles -------------------------------
    particles = JustPIC.init_particles(
        backend, nxcell, max_xcell, min_xcell, grid_vel...
    )
    arrays = JustPIC.SubgridDiffusionCellArrays(particles)
    # Test they are allocated in the right backend
    @test arrays.ΔT_subgrid isa TA(backend)
    @test arrays.pT0.data isa TA(backend)
    @test arrays.pΔT.data isa TA(backend)
    @test arrays.dt₀.data isa TA(backend)

    # with `d = 0` the subgrid correction vanishes and the particles pick up
    # the resolved increment as it stands
    xvi_p = JustPIC.add_periodic_ghost_nodes.(xvi)
    xci_p = JustPIC.add_periodic_ghost_nodes.(xci)
    ΔT_const = FT(3.5)
    ΔT_grid = TA(backend)(fill(ΔT_const, size(particles.index)))
    active = Array(particles.index.data)

    T_vertex = TA(backend)([z for x in xvi_p[1], y in xvi_p[2], z in xvi_p[3]])
    pT, = JustPIC.init_cell_arrays(particles, Val(1))
    JustPIC.grid2particle!(pT, T_vertex, particles)
    pT_before = copy(Array(pT.data))
    subgrid_diffusion!(pT, T_vertex, ΔT_grid, arrays, particles, FT(1); d = FT(0))
    @test Array(pT.data)[active] ≈ (pT_before .+ ΔT_const)[active]

    arrays_c = JustPIC.SubgridDiffusionCellArrays(particles; loc = :center)
    T_centroid = TA(backend)([z for x in xci_p[1], y in xci_p[2], z in xci_p[3]])
    pTc, = JustPIC.init_cell_arrays(particles, Val(1))
    JustPIC.centroid2particle!(pTc, T_centroid, particles)
    pTc_before = copy(Array(pTc.data))
    subgrid_diffusion_centroid!(pTc, T_centroid, ΔT_grid, arrays_c, particles, FT(1); d = FT(0))
    @test Array(pTc.data)[active] ≈ (pTc_before .+ ΔT_const)[active]

    @test_throws "ΔT_grid must carry one ghost node per side" subgrid_diffusion!(
        pT, T_vertex, TA(backend)(zeros(FT, ni...)), arrays, particles, FT(1)
    )
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
    i, j, k = JustPIC.cell_index(p, xv)
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
    i, j, k = JustPIC.cell_index(p, xv)
    @test x[i] ≤ px < x[i + 1]
    @test y[j] ≤ py < y[j + 1]
    @test z[k] ≤ pz < z[k + 1]
    GC.gc()
end

@testset "Periodic ghost nodes 3D" begin
    zv_uniform = LinRange(FT(-1), FT(1), 5)
    zv_uniform_periodic = JustPIC.add_periodic_ghost_nodes(zv_uniform)
    @test zv_uniform_periodic isa LinRange
    @test Array(zv_uniform_periodic) ≈ [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]

    zv_refined = FT[-1.0, -0.7, -0.2, 0.4, 1.0]
    zv_refined_periodic = JustPIC.add_periodic_ghost_nodes(zv_refined)
    @test zv_refined_periodic isa Vector
    @test zv_refined_periodic ≈ FT[-1.6, -1.0, -0.7, -0.2, 0.4, 1.0, 1.3]

    xv = LinRange(FT(0), FT(1), 5)
    xc = LinRange(FT(0.125), FT(0.875), 4)
    grid_vx = xv, expand_range(xc), expand_range(xc)
    grid_vy = expand_range(xc), xv, expand_range(xc)
    grid_vz = expand_range(xc), expand_range(xc), xv
    particles = init_particles(backend, 8, 12, 4, grid_vx, grid_vy, grid_vz)
    field, = init_cell_arrays(particles, Val(1))

    function reset_particle!(position)
        fill!(particles.index.data, false)
        foreach(coord -> fill!(coord.data, FT(NaN)), particles.coords)
        fill!(field.data, FT(NaN))
        CAI.@index particles.index[1, 5, 3, 5] = true
        for i in 1:3
            CAI.@index particles.coords[i][1, 5, 3, 5] = position[i]
        end
        CAI.@index field[1, 5, 3, 5] = FT(42)
        return nothing
    end

    reset_particle!((FT(1.01), FT(0.375), FT(1.02)))
    move_particles!(particles, (field,); periodic_1 = true, periodic_3 = true)
    @test count(Array(particles.index.data)) == 1
    @test CAI.@index(particles.index[1, 2, 3, 2])
    @test CAI.@index(particles.coords[1][1, 2, 3, 2]) ≈ FT(0.01)
    @test CAI.@index(particles.coords[3][1, 2, 3, 2]) ≈ FT(0.02)
    @test CAI.@index(field[1, 2, 3, 2]) == FT(42)

    Vx = TA(backend)(fill(FT(1), length.(grid_vx)))
    Vy = TA(backend)(fill(FT(0), length.(grid_vy)))
    Vz = TA(backend)(fill(FT(1), length.(grid_vz)))
    V = Vx, Vy, Vz
    for advection_fn in (advection!, advection_LinP!, advection_MQS!)
        for method in (Euler(), RungeKutta2(), RungeKutta4())
            reset_particle!((FT(0.99), FT(0.375), FT(0.99)))
            advection_fn(
                particles, method, V, FT(0.02);
                periodic_1 = true, periodic_3 = true,
            )
            move_particles!(particles, (field,); periodic_1 = true, periodic_3 = true)
            @test count(Array(particles.index.data)) == 1
            @test CAI.@index(particles.index[1, 2, 3, 2])
            @test CAI.@index(particles.coords[1][1, 2, 3, 2]) ≈ FT(0.01)
            @test CAI.@index(particles.coords[3][1, 2, 3, 2]) ≈ FT(0.01)
            @test CAI.@index(field[1, 2, 3, 2]) == FT(42)
        end
    end
end

include(joinpath(@__DIR__, "helpers_move_particles.jl"))

@testset "Particle movement fills free slots per destination 3D" begin
    xv = LinRange(FT(0), FT(1), 5)
    xc = LinRange(FT(0.125), FT(0.875), 4)
    grid_vx = xv, expand_range(xc), expand_range(xc)
    grid_vy = expand_range(xc), xv, expand_range(xc)
    grid_vz = expand_range(xc), expand_range(xc), xv
    particles = init_particles(backend, 8, 8, 1, grid_vx, grid_vy, grid_vz)
    fields = init_cell_arrays(particles, Val(2))

    xm, xp, ym, yp, zm, zp = (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)
    scenarios = (
        "two destinations" => ([xm, xp], Dict(xm => [8], xp => [1])),
        "six destinations" => (
            [xm, xp, ym, yp, zm, zp],
            Dict(xm => [8], xp => [1], ym => [2], yp => [7], zm => [3], zp => [6]),
        ),
        "alternating destinations" => ([xm, xp, xm, xp], Dict(xm => [7, 8], xp => [1, 2])),
        "shared destination and a staying particle" => ([xp, xm, xp, (0, 0, 0)], Dict(xp => [2, 3], xm => [1])),
        "corner destinations" => (
            [(1, 1, 1), (-1, -1, -1), (1, -1, 1), (-1, 1, -1)],
            Dict((1, 1, 1) => [8], (-1, -1, -1) => [1], (1, -1, 1) => [4], (-1, 1, -1) => [5]),
        ),
    )
    for (name, (leaving, free)) in scenarios
        @testset "$name" begin
            check_fragmented_move(backend, particles, fields, (3, 3, 3), leaving, free)
        end
    end

    @testset "full destination reports overflow" begin
        check_full_destination_overflow(particles, fields)
    end

    @testset "clean removes out-of-cell particles without compaction" begin
        check_clean_particles(particles, fields)
    end
end

include(joinpath(@__DIR__, "helpers_move_jumps.jl"))

@testset "Particle movement with converging jumps 3D" begin
    ncells = (14, 12, 10)
    xv, yv, zv = ntuple(d -> LinRange(FT(0), FT(1), ncells[d] + 1), Val(3))
    xc, yc, zc = ntuple(Val(3)) do d
        v = (xv, yv, zv)[d]
        LinRange(v[1] + step(v) / 2, v[end] - step(v) / 2, ncells[d])
    end
    grid_vx = xv, expand_range(yc), expand_range(zc)
    grid_vy = expand_range(xc), yv, expand_range(zc)
    grid_vz = expand_range(xc), expand_range(yc), zv

    for max_jump in (1, 2, (2, 0, 1)), periodicity in ((false, false, false), (true, false, true), (true, true, true))
        particles = init_particles(backend, 8, 40, 4, grid_vx, grid_vy, grid_vz)
        args = init_cell_arrays(particles, Val(2))
        expected = load_jumping_particles!(particles, args, max_jump, periodicity)

        move_particles!(
            particles, args;
            periodic_1 = periodicity[1], periodic_2 = periodicity[2], periodic_3 = periodicity[3],
        )

        report = audit_jumping_particles(particles, args, expected)
        @test report == (; found = length(expected), duplicated = 0, misplaced = 0, corrupted = 0, lost = 0)
    end
end

@testset "Refined grid particle initialization 3D" begin
    xv = FT[0.0, 0.1, 0.25, 0.55, 1.0]
    yv = FT[0.0, 0.2, 0.45, 0.8, 1.0]
    zv = FT[0.0, 0.15, 0.35, 0.65, 1.0]
    xvi = (xv, yv, zv)
    xci = ntuple(i -> [(xvi[i][j] + xvi[i][j + 1]) / 2 for j in 1:(length(xvi[i]) - 1)], Val(3))
    grid_vi = (
        (xv, expand_range(xci[2]), expand_range(xci[3])),
        (expand_range(xci[1]), yv, expand_range(xci[3])),
        (expand_range(xci[1]), expand_range(xci[2]), zv),
    )

    nxcell, max_xcell, min_xcell = 8, 12, 4
    particles = JustPIC.init_particles(
        backend, nxcell, max_xcell, min_xcell, grid_vi...,
    )

    @test Array.(particles.xvi) == JustPIC.add_periodic_ghost_nodes.(xvi)
    @test Array.(particles.xci) == JustPIC.add_periodic_ghost_nodes.(xci)
    @test Array.(particles.xi_vel[1]) == grid_vi[1]
    @test Array.(particles.xi_vel[2]) == grid_vi[2]
    @test Array.(particles.xi_vel[3]) == grid_vi[3]
end

@testset "Passive markers 3D" begin
    n = 32
    nx = ny = nz = n - 1
    Lx = Ly = Lz = FT(1)
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
    particles = JustPIC.init_particles(
        backend, nxcell, max_xcell, min_xcell, grid_vel...
    )

    # Cell fields -------------------------------
    Vx = TA(backend)([vx_stream_3D(x, z) for x in grid_vx[1], y in grid_vx[2], z in grid_vx[3]])
    Vy = TA(backend)([vy_stream_3D(x, z) for x in grid_vy[1], y in grid_vy[2], z in grid_vy[3]])
    Vz = TA(backend)([vz_stream_3D(x, z) for x in grid_vz[1], y in grid_vz[2], z in grid_vz[3]])
    T = TA(backend)([z for x in xv, y in yv, z in zv])
    P = TA(backend)([x for x in xv, y in yv, z in zv])
    V = Vx, Vy, Vz

    dt = min(dx / maximum(abs.(Vx)), dy / maximum(abs.(Vy)), dz / maximum(abs.(Vz))) / 2
    np = 256 # number of passive markers

    passive_coords = ntuple(Val(3)) do i
        TA(backend)((rand(FT, np) .+ 1) .* Lx / 4)
    end

    passive_markers = JustPIC.init_passive_markers(backend, passive_coords)
    T_marker = TA(backend)(zeros(FT, np))
    P_marker = TA(backend)(zeros(FT, np))

    for _ in 1:75
        JustPIC.advection!(passive_markers, JustPIC.RungeKutta2(2 / 3), V, (grid_vx, grid_vy, grid_vz), dt)
    end

    # interpolate grid fields T and P onto the marker locations
    JustPIC.grid2particle!((T_marker, P_marker), xvi, (T, P), passive_markers)
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
    Lx = Ly = Lz = FT(1)
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

    particles = JustPIC.init_particles(backend, nxcell, max_xcell, min_xcell, grid_vel...)
    pphase, = JustPIC.init_cell_arrays(particles, Val(1))

    ni = size(particles.index)
    nslots = JustPIC.cellnum(particles.index)
    p_invalid = ForceInjectionPoint3D((FT(0), FT(0), FT(0)), false)
    p_new = TA(backend)(fill(p_invalid, ni..., nslots))
    for i in 1:ni[1], j in 1:ni[2], k in 1:ni[3], c in 1:nslots
        p_new[i, j, k, c] = ForceInjectionPoint3D((FT(0.1) * c + FT(0.01) * i, FT(0.1) * c + FT(0.01) * j, FT(0.1) * c + FT(0.01) * k), true)
    end

    JustPIC.force_injection!(particles, p_new, (pphase,), (FT(5),))

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

    particles_no_fields = JustPIC.init_particles(backend, nxcell, max_xcell, min_xcell, grid_vel...)
    JustPIC.force_injection!(particles_no_fields, p_new)
    @test all(Array(particles_no_fields.index.data))

    particles_skip = JustPIC.init_particles(backend, nxcell, max_xcell, min_xcell, grid_vel...)
    p_empty = TA(backend)(fill(p_invalid, ni..., nslots))
    JustPIC.force_injection!(particles_skip, p_empty)
    @test !any(Array(particles_skip.index.data))

    particles_partial = JustPIC.init_particles(backend, nxcell, max_xcell, min_xcell, grid_vel...)
    p_partial = TA(backend)(fill(p_invalid, ni..., nslots))
    p_partial[1, 1, 1, 1] = ForceInjectionPoint3D((FT(0.2), FT(0.3), FT(0.4)), true)
    JustPIC.force_injection!(particles_partial, p_partial)
    active = Array(particles_partial.index.data)
    @test count(active) == 1
    @test all(isfinite, Array(particles_partial.coords[1].data)[active])
    @test all(isfinite, Array(particles_partial.coords[2].data)[active])
    @test all(isfinite, Array(particles_partial.coords[3].data)[active])
end

function _assert_finite_advection_3D!(particles, field, step, layout)
    active = Array(particles.index.data)
    for (dim, coords) in enumerate(particles.coords)
        values = Array(coords.data)
        bad = findfirst(I -> active[I] && !isfinite(values[I]), CartesianIndices(values))
        isnothing(bad) || error(
            "3D advection produced invalid particle coordinate: " *
                "backend=$(BACKEND_NAME), precision=$(FT), seed=0, " *
                "layout=$layout, step=$step, dimension=$dim, index=$bad",
        )
    end
    values = Array(field)
    bad = findfirst(x -> !isfinite(x), values)
    isnothing(bad) || error(
        "3D advection produced invalid field value: " *
            "backend=$(BACKEND_NAME), precision=$(FT), seed=0, " *
            "layout=$layout, step=$step, index=$bad",
    )
    return nothing
end

function _particle_layout_signature_3D(particles, label)
    active = Array(particles.index.data)
    first_active = findfirst(identity, active)
    first_point = ntuple(dim -> Array(particles.coords[dim].data)[first_active], Val(3))
    return "$label, active_count=$(count(active)), first_active=$first_active, first_point=$first_point"
end

function _nodal_weights_3D(x)
    weights = Vector{eltype(x)}(undef, length(x))
    weights[1] = (x[2] - x[1]) / 2
    weights[end] = (x[end] - x[end - 1]) / 2
    for i in 2:(length(x) - 1)
        weights[i] = (x[i + 1] - x[i - 1]) / 2
    end
    return weights
end

function _weighted_integral_3D(field, xvi)
    values = Array(field)[2:(end - 1), 2:(end - 1), 2:(end - 1)]
    wx, wy, wz = _nodal_weights_3D.(xvi)
    return sum(
        values .* reshape(wx, :, 1, 1) .* reshape(wy, 1, :, 1) .* reshape(wz, 1, 1, :),
    )
end

function test_advection_3D()

    n = 64
    nx = ny = nz = n - 1
    Lx = Ly = Lz = FT(1)
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
    Vx = TA(backend)([vx_stream_3D(x, z) for x in grid_vx[1], y in grid_vx[2], z in grid_vx[3]])
    Vy = TA(backend)([vy_stream_3D(x, z) for x in grid_vy[1], y in grid_vy[2], z in grid_vy[3]])
    Vz = TA(backend)([vz_stream_3D(x, z) for x in grid_vz[1], y in grid_vz[2], z in grid_vz[3]])
    xvi_p = JustPIC.add_periodic_ghost_nodes.(xvi)
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
    particles = JustPIC.init_particles(
        backend, nxcell, max_xcell, min_xcell, grid_vel...
    )
    layout = _particle_layout_signature_3D(
        particles, "regular(n=$n, nxcell=$nxcell, max_xcell=$max_xcell, min_xcell=$min_xcell)",
    )

    # Advection test
    particle_args = pT, = JustPIC.init_cell_arrays(particles, Val(1))
    JustPIC.grid2particle!(pT, xvi_p, T, particles, diff.(xvi_p))
    # Particle-to-grid interpolation is nonconservative; weighted integral checks bounded drift.
    sumT = _weighted_integral_3D(T, xvi)

    niter = 5
    for step in 1:niter
        JustPIC.particle2grid!(T, pT, particles)
        copyto!(T0, T)
        JustPIC.advection!(particles, JustPIC.RungeKutta2(), V, dt)
        JustPIC.move_particles!(particles, particle_args)
        # reseed
        JustPIC.inject_particles!(particles, (pT,))
        JustPIC.grid2particle!(pT, xvi_p, T, particles, diff.(xvi_p))
        _assert_finite_advection_3D!(
            particles, T, step, layout,
        )
    end
    sumT_final = _weighted_integral_3D(T, xvi)
    err = abs(sumT - sumT_final) / sumT
    println(err)
    return err
end

function test_advection_3D_refined()
    xv = FT[0.0, 0.04, 0.09, 0.16, 0.25, 0.37, 0.52, 0.68, 0.84, 1.0]
    yv = FT[0.0, 0.05, 0.11, 0.2, 0.31, 0.44, 0.58, 0.73, 0.87, 1.0]
    zv = FT[0.0, 0.03, 0.08, 0.15, 0.26, 0.4, 0.56, 0.72, 0.88, 1.0]
    xvi = (xv, yv, zv)
    xc = [(xv[i] + xv[i + 1]) / 2 for i in 1:(length(xv) - 1)]
    yc = [(yv[i] + yv[i + 1]) / 2 for i in 1:(length(yv) - 1)]
    zc = [(zv[i] + zv[i + 1]) / 2 for i in 1:(length(zv) - 1)]

    grid_vx = xv, expand_range(yc), expand_range(zc)
    grid_vy = expand_range(xc), yv, expand_range(zc)
    grid_vz = expand_range(xc), expand_range(yc), zv

    Vx = TA(backend)([vx_stream_3D(x, z) for x in grid_vx[1], y in grid_vx[2], z in grid_vx[3]])
    Vy = TA(backend)([vy_stream_3D(x, z) for x in grid_vy[1], y in grid_vy[2], z in grid_vy[3]])
    Vz = TA(backend)([vz_stream_3D(x, z) for x in grid_vz[1], y in grid_vz[2], z in grid_vz[3]])
    xvi_p = JustPIC.add_periodic_ghost_nodes.(xvi)
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
    particles = JustPIC.init_particles(
        backend, nxcell, max_xcell, min_xcell, grid_vx, grid_vy, grid_vz,
    )
    layout = _particle_layout_signature_3D(
        particles, "refined(nxcell=$nxcell, max_xcell=$max_xcell, min_xcell=$min_xcell)",
    )

    particle_args = pT, = JustPIC.init_cell_arrays(particles, Val(1))
    JustPIC.grid2particle!(pT, xvi_p, T, particles, diff.(xvi_p))
    # Refined-grid check remains a bounded-drift check for nonconservative interpolation.
    sumT = _weighted_integral_3D(T, xvi)

    niter = 5
    for step in 1:niter
        JustPIC.particle2grid!(T, pT, particles)
        copyto!(T0, T)
        JustPIC.advection!(particles, JustPIC.RungeKutta2(), V, dt)
        JustPIC.move_particles!(particles, particle_args)
        JustPIC.inject_particles!(particles, (pT,))
        JustPIC.grid2particle!(pT, xvi_p, T, particles, diff.(xvi_p))
        _assert_finite_advection_3D!(
            particles, T, step, layout,
        )
    end

    sumT_final = _weighted_integral_3D(T, xvi)
    err = abs(sumT - sumT_final) / sumT
    println(err)
    return err
end

function test_advection()
    err = test_advection_3D()
    tol = 1.0e-1
    passed = err < tol
    return passed
end

function test_advection_refined()
    err = test_advection_3D_refined()
    tol = 1.0e-1
    passed = err < tol
    return passed
end

@testset "Miniapps" begin
    @test test_advection()
    @test test_advection_refined()
end

@static if ENV["JULIA_JUSTPIC_BACKEND"] === "AMDGPU"
    using AMDGPU
    AMDGPU.allowscalar(true)
elseif ENV["JULIA_JUSTPIC_BACKEND"] === "CUDA"
    using CUDA
    CUDA.allowscalar(true)
end

using JustPIC, JustPIC._2D, CellArrays, Test, LinearAlgebra

const backend = @static if ENV["JULIA_JUSTPIC_BACKEND"] === "AMDGPU"
    AMDGPUBackend
elseif ENV["JULIA_JUSTPIC_BACKEND"] === "CUDA"
    CUDABackend
else
    CPUBackend
end

function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = round(x1-dx; sigdigits=5)
    xF = round(x2+dx; sigdigits=5)
    LinRange(xI, xF, n+2)
end

# Analytical flow solution
vx_stream(x, y) =  250 * sin(π*x) * cos(π*y)
vy_stream(x, y) = -250 * cos(π*x) * sin(π*y)

# Analytical flow solution
vi_stream(x) =  π * 1e-5 * (x - 0.5)

@testset "Interpolations 2D" begin
    nxcell, max_xcell, min_xcell = 24, 24, 1
    n = 5 # number of vertices
    nx = ny = n - 1
    Lx = Ly = 1.0
    # nodal vertices
    xvi = xv, yv = LinRange(0, Lx, n), LinRange(0, Ly, n)
    dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
    # nodal centers
    xci = xc, yc = LinRange(0+dx/2, Lx-dx/2, n-1), LinRange(0+dy/2, Ly-dy/2, n-1)
    # staggered grid velocity nodal locations

    # Initialize particles & particle fields
    particles = _2D.init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi..., dxi..., nx, ny
    )
    pT, = _2D.init_cell_arrays(particles, Val(1))

    # Linear field at the vertices
    T  = TA(backend)([y for x in xv, y in yv])
    T0 = TA(backend)([y for x in xv, y in yv])

    # Grid to particle test
    _2D.grid2particle!(pT, xvi, T, particles)

    @test pT == particles.coords[2]

    # Grid to particle test
    _2D.grid2particle_flip!(pT, xvi, T, T0, particles)

    @test pT == particles.coords[2]

    # Particle to grid test
    T2 = similar(T)
    _2D.particle2grid!(T2, pT, xvi, particles)
    # norm(T2 .- T) / length(T)
    @test norm(T2 .- T) / length(T) < 1e-1
end

@testset "Subgrid diffusion 2D" begin
    nxcell, max_xcell, min_xcell = 12, 12, 1
    n = 5 # number of vertices
    nx = ny = n - 1
    Lx = Ly = 1.0
    # nodal vertices
    xvi = xv, yv = LinRange(0, Lx, n), LinRange(0, Ly, n)
    dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]

    # Initialize particles & particle fields
    particles = _2D.init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi..., dxi..., nx, ny
    )

    arrays = SubgridDiffusionCellArrays(particles)
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
    nx = ny = n-1
    ni = nx, ny
    Lx = Ly = 1.0
    # nodal vertices
    xvi = xv, yv = LinRange(0, Lx, n), LinRange(0, Ly, n)
    dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
    # nodal centers
    xc, yc = LinRange(0+dx/2, Lx-dx/2, n-1), LinRange(0+dy/2, Ly-dy/2, n-1)
    # staggered grid velocity nodal locations
    grid_vx = xv, expand_range(yc)
    grid_vy = expand_range(xc), yv

    particles1 = _2D.init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi..., dxi..., ni...
    )

    particles2 = _2D.init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi, dxi, ni
    )

    @test particles1.min_xcell == particles2.min_xcell
    @test particles1.max_xcell == particles2.max_xcell
    @test particles1.np == particles2.np
end

@testset "Cell index 2D" begin
    n = 11
    x = range(0, stop=1, length=n)
    xv = x, x
    
    px = rand()
    idx = _2D.cell_index(px, x)
    @test x[idx] ≤ px < x[idx+1]
    
    px, py = rand(2)
    i, j = _2D.cell_index((px,py), xv)
    @test x[i] ≤ px < x[i+1]
    @test x[j] ≤ py < x[j+1]
    
    x = range(0, stop=1, length=n)
    y = range(-1, stop=0, length=n)
    px, py = rand(), -rand()
    idx = cell_index(py, y)
    @test y[idx] ≤ py < y[idx+1]
    
    xv = x, y
    i, j = _2D.cell_index((px,py), xv)
    @test x[i] ≤ px < x[i+1]
    @test y[j] ≤ py < y[j+1] 
end

@testset "Passive markers 2D" begin
    # Initialize particles -------------------------------
    n = 51
    Lx = Ly = 1.0
    # nodal vertices
    xvi = xv, yv = LinRange(0, Lx, n), LinRange(0, Ly, n)
    dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
    # nodal centers
    xc, yc = LinRange(0+dx/2, Lx-dx/2, n-1), LinRange(0+dy/2, Ly-dy/2, n-1)
    # staggered grid velocity nodal locations
    grid_vx = xv, expand_range(yc)
    grid_vy = expand_range(xc), yv

    # Cell fields -------------------------------
    Vx = TA(backend)([-vi_stream(y) for x in grid_vx[1], y in grid_vx[2]]);
    Vy = TA(backend)([ vi_stream(x) for x in grid_vy[1], y in grid_vy[2]]);

    T   = TA(backend)([y for x in xv, y in yv]);
    P   = TA(backend)([x for x in xv, y in yv]);
    V   = Vx, Vy;

    w = π*1e-5  # angular velocity
    period = 1  # revolution number
    tmax = period / (w/(2*π))
    dt = 200.0

    np = 256 # number of passive markers
    passive_coords = ntuple(Val(2)) do i
        TA(backend)((rand(np) .+ 1) .* Lx/4)
    end

    passive_markers = init_passive_markers(backend, passive_coords);
    T_marker = TA(backend)(zeros(np))
    P_marker = TA(backend)(zeros(np))

    for _ in 1:50
        _2D.advection!(passive_markers, RungeKutta2(2/3), V, (grid_vx, grid_vy), dt)
    end

    # interpolate grid fields T and P onto the marker locations
    _2D.grid2particle!((T_marker, P_marker), xvi, (T, P), passive_markers)
    x_marker = passive_markers.coords[1]
    y_marker = passive_markers.coords[2]

    @test y_marker ≈ T_marker
    @test x_marker ≈ P_marker
end

function advection_test_2D()
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 6, 12, 1
    n = 64
    nx = ny = n-1
    Lx = Ly = 1.0
    # nodal vertices
    xvi = xv, yv = LinRange(0, Lx, n), LinRange(0, Ly, n)
    dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
    # nodal centers
    xci = xc, yc = LinRange(0+dx/2, Lx-dx/2, n-1), LinRange(0+dy/2, Ly-dy/2, n-1)
    # staggered grid velocity nodal locations
    grid_vx = xv, expand_range(yc)
    grid_vy = expand_range(xc), yv

    particles = _2D.init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi..., dxi..., nx, ny
    )

    # Cell fields -------------------------------
    Vx = TA(backend)([vx_stream(x, y) for x in grid_vx[1], y in grid_vx[2]]);
    Vy = TA(backend)([vy_stream(x, y) for x in grid_vy[1], y in grid_vy[2]]);
    T  = TA(backend)([y for x in xv, y in yv]);
    T0 = deepcopy(T)
    V  = Vx, Vy;

    dt = min(dx / maximum(abs.(Array(Vx))),  dy / maximum(abs.(Array(Vy)))) / 2;

    # Advection test
    particle_args = pT, = init_cell_arrays(particles, Val(1));
    _2D.grid2particle!(pT, xvi, T, particles);

    sumT = sum(T)

    niter = 25
    for it in 1:niter
        _2D.particle2grid!(T, pT, xvi, particles)
        copyto!(T0, T)
        _2D.advection!(particles, RungeKutta2(2/3), V, (grid_vx, grid_vy), dt)
        _2D.move_particles!(particles, xvi, particle_args)

        inject = _2D.check_injection(particles)
        inject && _2D.inject_particles!(particles, (pT, ), (T,), xvi)

        _2D.grid2particle!(pT, xvi, T, particles)
    end

    sumT_final = sum(T)

    return abs(sumT - sumT_final) / sumT

end

function test_advection_2D()
    err = advection_test_2D()
    tol = 1e-2
    passed = err < tol

    return passed
end

function test_rotating_circle()
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 12, 24, 6
    n = 101
    nx = ny = n-1
    Lx = Ly = 1.0
    # nodal vertices
    xvi = xv, yv = LinRange(0, Lx, n), LinRange(0, Ly, n)
    dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
    # nodal centers
    xc, yc = LinRange(0+dx/2, Lx-dx/2, n-1), LinRange(0+dy/2, Ly-dy/2, n-1)
    # staggered grid velocity nodal locations
    grid_vx = xv, expand_range(yc)
    grid_vy = expand_range(xc), yv

    particles = _2D.init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi..., dxi..., nx, ny
    )

    # Cell fields -------------------------------
    Vx = TA(backend)([-vi_stream(y) for x in grid_vx[1], y in grid_vx[2]]);
    Vy = TA(backend)([ vi_stream(x) for x in grid_vy[1], y in grid_vy[2]]);
    xc0 = yc0 =  0.25
    R   = 6 * dx
    T   = TA(backend)([((x-xc0)^2 + (y-yc0)^2 ≤ R^2)  * 1.0 for x in xv, y in yv]);
    T0  = deepcopy(T)
    V   = Vx, Vy;

    w      = π * 1e-5  # angular velocity
    period = 1  # revolution number
    tmax   = period / (w/(2*π)) / 10
    dt     = 200.0

    particle_args = pT, = init_cell_arrays(particles, Val(1));
    _2D.grid2particle!(pT, xvi, T, particles);

    t = 0
    it = 0
    sumT = sum(T)
    while t ≤ tmax
        _2D.particle2grid!(T, pT, xvi, particles)
        copyto!(T0, T)
        _2D.advection!(particles, RungeKutta2(2/3), V, (grid_vx, grid_vy), dt)
        _2D.move_particles!(particles, xvi, particle_args)

        inject = _2D.check_injection(particles)
        inject && _2D.inject_particles!(particles, (pT, ), (T,), xvi)
        _2D.grid2particle!(pT, xvi, T, particles)
        t += dt
        it += 1
    end

    sumT_final = sum(T)

    return abs(sumT - sumT_final) / sumT
end

function test_rotation_2D()
    err = test_rotating_circle()
    tol = 1e-2
    passed = err < tol

    return passed
end

@testset "Miniapps" begin
    @testset "1. Advection 2D" begin
        @test test_advection_2D()
    end

    @testset "2. Rotating circle 2D" begin
        @test test_rotation_2D()
    end
end

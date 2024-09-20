
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
    nxcell, max_xcell, min_xcell = 5, 5, 1
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
        backend, nxcell, max_xcell, min_xcell, xvi...,
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

    # test Array conversion
    particles_cpu = Array(particles)
    pT_cpu        = Array(pT)
    @test particles_cpu.index isa JustPIC.CellArrays.CPUCellArray
    @test pT_cpu              isa JustPIC.CellArrays.CPUCellArray
    @test particles_cpu.index.data[:] == Array(particles.index.data)[:]
    @test pT_cpu.data[:]              == Array(pT.data)[:]

    # test copy function
    particles_copy = copy(particles)
    pT_copy        = copy(pT)
    @test particles_copy.index.data[:] == particles.index.data[:]
    @test pT_copy.data[:]              == pT.data[:]
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
        backend, nxcell, max_xcell, min_xcell, xvi...,
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
        backend, nxcell, max_xcell, min_xcell, xvi...,
    )

    particles2 = _2D.init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi...,
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

@testset "Pure shear 2D" begin
    
    @parallel_indices (I...) function InitialFieldsParticles!(phases, px, py, index)
         for ip in cellaxes(phases)
            # quick escape
            @index(index[ip, I...]) == 0 && continue
            x = @index px[ip, I...]
            y = @index py[ip, I...]
            if x<y
                @index phases[ip, I...] = 1.0
            else
                @index phases[ip, I...] = 2.0
            end
        end
        return nothing
    end

    year = 365*3600*24
    L    = (x=1., y=1.)
    Nc   = (x=41, y=41 )
    Nv   = (x=Nc.x+1,   y=Nc.y+1   )
    Δ    = (x=L.x/Nc.x, y=L.y/Nc.y )
    Nt   = 200
    Nout = 1
    C    = 0.25

    verts     = (x=LinRange(-L.x/2, L.x/2, Nv.x), y=LinRange(-L.y/2, L.y/2, Nv.y))
    cents     = (x=LinRange(-Δ.x/2+L.x/2, L.x/2-Δ.x/2, Nc.x), y=LinRange(-Δ.y/2+L.y/2, L.y+Δ.y/2-L.y/2, Nc.y))
    cents_ext = (x=LinRange(-Δ.x/2-L.x/2, L.x/2+Δ.x/2, Nc.x+2), y=LinRange(-Δ.y/2-L.y/2, L.y+Δ.y/2+L.y/2, Nc.y+2))

    size_x = (Nc.x+1, Nc.y+2)
    size_y = (Nc.x+2, Nc.y+1)

    V = (
        x      = @zeros(size_x),
        y      = @zeros(size_y),
    )

    # Set velocity field
    ε̇bg = -1.0
    for i=1:size(V.x,1),  j=1:size(V.x,2)
        V.x[i,j] =  verts.x[i]*ε̇bg
    end

    for i=1:size(V.y,1),  j=1:size(V.y,2)
        V.y[i,j] = -verts.y[j]*ε̇bg
    end
 
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 12, 24, 5
    particles = init_particles(
        backend, 
        nxcell, 
        max_xcell,
        min_xcell, 
        values(verts),
        values(Δ),
        values(Nc)
    ) # random position by default

    # Initialise phase field
    particle_args = phases, = init_cell_arrays(particles, Val(1))  # cool

    @parallel InitialFieldsParticles!(phases, particles.coords..., particles.index)

    phase_ratios = JustPIC._2D.PhaseRatios(backend, 2, values(Nc));
    phase_ratios_vertex!(phase_ratios, particles, values(verts), phases) 
    phase_ratios_center!(phase_ratios, particles, values(verts), phases) 

    # Time step
    t  = 0e0
    Δt = C * min(Δ...) / max(maximum(abs.(V.x)), maximum(abs.(V.y)))

    # Create necessary tuples
    grid_vx = (verts.x, cents_ext.y)
    grid_vy = (cents_ext.x, verts.y)
    Vxc     = 0.5*(V.x[1:end-1,2:end-1] .+ V.x[2:end-0,2:end-1])
    Vyc     = 0.5*(V.y[2:end-1,1:end-1] .+ V.y[2:end-1,2:end-0])

    for it=1:Nt
        advection_MQS!(particles, RungeKutta2(), values(V), (grid_vx, grid_vy), Δt)
        move_particles!(particles, values(verts), particle_args)        
        inject_particles!(particles, particle_args, values(verts)) 

        phase_ratios_vertex!(phase_ratios, particles, values(verts), phases) 
        phase_ratios_center!(phase_ratios, particles, values(verts), phases)     
        min_prv, max_prv = extrema(sum(phase_ratios.vertex.data, dims=2))
        min_prc, max_prc = extrema(sum(phase_ratios.center.data, dims=2))
        
        @test min_prv ≈ 1
        @test max_prv ≈ 1
        @test min_prc ≈ 1
        @test max_prc ≈ 1
    end
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
        backend, nxcell, max_xcell, min_xcell, xvi...,
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
        _2D.inject_particles!(particles, (pT, ), xvi)
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
    nxcell, max_xcell, min_xcell = 50, 60, 40
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
        backend, nxcell, max_xcell, min_xcell, xvi...,
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
        _2D.advection!(particles, _2D.RungeKutta2(2/3), V, (grid_vx, grid_vy), dt)
        _2D.move_particles!(particles, xvi, particle_args)
        _2D.inject_particles!(particles, (pT, ), xvi)
        _2D.grid2particle!(pT, xvi, T, particles)
        t  += dt
        it += 1
    end

    sumT_final = sum(T)

    return abs(sumT - sumT_final) / sumT
end

function test_rotation_2D()
    err = test_rotating_circle()
    tol = 1e-1
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

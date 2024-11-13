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

function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = round(x1-dx; sigdigits=5)
    xF = round(x2+dx; sigdigits=5)
    LinRange(xI, xF, n+2)
end

# Analytical flow solution
vx_stream(x, z) =  250 * sin(π*x) * cos(π*z)
vy_stream(x, z) =  0.0
vz_stream(x, z) = -250 * cos(π*x) * sin(π*z)

@testset "Interpolations 3D" begin
    nxcell, max_xcell, min_xcell = 12, 12, 1
    n   = 5 # number of vertices
    nx  = ny = nz = n-1
    ni  = nx, ny, nz
    Lx  = Ly = Lz = 1.0
    Li  = Lx, Ly, Lz
    # nodal vertices
    xvi = xv, yv, zv = ntuple(i -> LinRange(0, Li[i], n), Val(3))
    # grid spacing
    dxi = dx, dy, dz = ntuple(i -> xvi[i][2] - xvi[i][1], Val(3))
    # nodal centers
    xci = xc, yc, zc = ntuple(i -> LinRange(0+dxi[i]/2, Li[i]-dxi[i]/2, ni[i]), Val(3))
    # Initialize particles -------------------------------
    particles = _3D.init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi...
    )
    pT, = _3D.init_cell_arrays(particles, Val(1))
    # Linear field at the vertices
    T  = TA(backend)([z for x in xv, y in yv, z in zv])
    T0 = TA(backend)([z for x in xv, y in yv, z in zv])
    # Grid to particle test
    _3D.grid2particle!(pT, xvi, T, particles)
    @test pT ≈ particles.coords[3]
    # Grid to particle test
    _3D.grid2particle_flip!(pT, xvi, T, T0, particles)
    @test pT ≈ particles.coords[3]
    # Particle to grid test
    T2 = similar(T)
    _3D.particle2grid!(T2, pT, xvi, particles)
    @test norm(T2 .- T) / length(T) < 1e-1
    # test copy function
    particles_copy = copy(particles)
    pT_copy        = copy(pT)
    @test particles_copy.index.data[:] == particles.index.data[:]
    @test pT_copy.data[:]              == pT.data[:]
end

@testset "Particles initialization 3D" begin
    nxcell, max_xcell, min_xcell = 24, 24, 1
    n   = 5 # number of vertices
    nx  = ny = nz = n-1
    ni  = nx, ny, nz
    Lx  = Ly = Lz = 1.0
    Li  = Lx, Ly, Lz
    # nodal vertices
    xvi = xv, yv, zv = ntuple(i -> LinRange(0, Li[i], n), Val(3))
    # grid spacing
    dxi = dx, dy, dz = ntuple(i -> xvi[i][2] - xvi[i][1], Val(3))
    # nodal centers
    xci = xc, yc, zc = ntuple(i -> LinRange(0+dxi[i]/2, Li[i]-dxi[i]/2, ni[i]), Val(3))
    # Initialize particles -------------------------------
    particles1 = _3D.init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi...
    )
    particles2 = _3D.init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi...
    )
    @test particles1.min_xcell == particles2.min_xcell
    @test particles1.max_xcell == particles2.max_xcell
    @test particles1.np == particles2.np
end

@testset "Subgrid diffusion 3D" begin
    nxcell, max_xcell, min_xcell = 24, 24, 1
    n   = 5 # number of vertices
    nx  = ny = nz = n-1
    ni  = nx, ny, nz
    Lx  = Ly = Lz = 1.0
    Li  = Lx, Ly, Lz
    # nodal vertices
    xvi = xv, yv, zv = ntuple(i -> LinRange(0, Li[i], n), Val(3))
    # grid spacing
    dxi = dx, dy, dz = ntuple(i -> xvi[i][2] - xvi[i][1], Val(3))
    # nodal centers
    xci = xc, yc, zc = ntuple(i -> LinRange(0+dxi[i]/2, Li[i]-dxi[i]/2, ni[i]), Val(3))
    # Initialize particles -------------------------------
    particles = _3D.init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi...
    )
    arrays = _3D.SubgridDiffusionCellArrays(particles)
    # Test they are allocated in the right backend
    @test arrays.ΔT_subgrid isa TA(backend)
    @test arrays.pT0.data isa TA(backend)
    @test arrays.pΔT.data isa TA(backend)
    @test arrays.dt₀.data isa TA(backend)
end

@testset "Cell index 3D" begin
    n = 100
    a, b = rand()*50, rand()*50
    start, finish = extrema((a, b))
    L = finish - start
    x = range(start, stop=finish, length=n)
    xv = x, x, x
    p = px, py, pz = tuple((rand(3).*L .+ start)...)
    i, j, k = _3D.cell_index(p, xv)
    @test x[i] ≤ px < x[i+1]
    @test x[j] ≤ py < x[j+1]
    @test x[k] ≤ pz < x[k+1]
    y = x
    z = range(-start, stop=finish, length=n)
    xv = x, y, z
    px, py = tuple((rand(2).*L .+ start)...)
    Lz = z[end] - z[1]
    pz = rand()*Lz - start
    p = px, py, pz
    i, j, k = _3D.cell_index(p, xv)
    @test x[i] ≤ px < x[i+1]
    @test y[j] ≤ py < y[j+1]
    @test z[k] ≤ pz < z[k+1]
end

@testset "Passive markers 3D" begin
    n   = 32
    nx  = ny = nz = n-1
    Lx  = Ly = Lz = 1.0
    ni  = nx, ny, nz
    Li  = Lx, Ly, Lz
    # nodal vertices
    xvi = xv, yv, zv = ntuple(i -> LinRange(0, Li[i], n), Val(3))
    # grid spacing
    dxi = dx, dy, dz = ntuple(i -> xvi[i][2] - xvi[i][1], Val(3))
    # nodal centers
    xci = xc, yc, zc = ntuple(i -> LinRange(0+dxi[i]/2, Li[i]-dxi[i]/2, ni[i]), Val(3))
    
    # staggered grid velocity nodal locations
    grid_vx = xv              , expand_range(yc), expand_range(zc)
    grid_vy = expand_range(xc), yv              , expand_range(zc)
    grid_vz = expand_range(xc), expand_range(yc), zv
    
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 24, 24, 3
    particles = _3D.init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi...
    )
    
    # Cell fields -------------------------------
    Vx = TA(backend)([vx_stream(x, z) for x in grid_vx[1], y in grid_vx[2], z in grid_vx[3]])
    Vy = TA(backend)([vy_stream(x, z) for x in grid_vy[1], y in grid_vy[2], z in grid_vy[3]])
    Vz = TA(backend)([vz_stream(x, z) for x in grid_vz[1], y in grid_vz[2], z in grid_vz[3]])
    T  = TA(backend)([z for x in xv, y in yv, z in zv])
    P  = TA(backend)([x for x in xv, y in yv, z in zv])
    V  = Vx, Vy, Vz
   
    dt = min(dx / maximum(abs.(Vx)), dy / maximum(abs.(Vy)), dz / maximum(abs.(Vz))) / 2
    np = 256 # number of passive markers
   
    passive_coords = ntuple(Val(3)) do i
        TA(backend)((rand(np) .+ 1) .* Lx/4)
    end
   
    passive_markers = _3D.init_passive_markers(backend, passive_coords);
    T_marker        = TA(backend)(zeros(np))
    P_marker        = TA(backend)(zeros(np))
   
    for _ in 1:75
        _3D.advection!(passive_markers, _3D.RungeKutta2(2/3), V, (grid_vx, grid_vy, grid_vz), dt)
    end
   
    # interpolate grid fields T and P onto the marker locations
    _3D.grid2particle!((T_marker, P_marker), xvi, (T, P), passive_markers)
    x_marker = passive_markers.coords[1]
    z_marker = passive_markers.coords[3]
   
    @test x_marker ≈ P_marker
    @test z_marker ≈ T_marker
end

function test_advection_3D()
      
    n   = 32
    nx  = ny = nz = n-1
    Lx  = Ly = Lz = 1.0
    ni  = nx, ny, nz
    Li  = Lx, Ly, Lz
    # nodal vertices
    xvi = xv, yv, zv = ntuple(i -> LinRange(0, Li[i], n), Val(3))
    # grid spacing
    dxi = dx, dy, dz = ntuple(i -> xvi[i][2] - xvi[i][1], Val(3))
    # nodal centers
    xci = xc, yc, zc = ntuple(i -> LinRange(0+dxi[i]/2, Li[i]-dxi[i]/2, ni[i]), Val(3))
    
    # staggered grid velocity nodal locations
    grid_vx = xv              , expand_range(yc), expand_range(zc)
    grid_vy = expand_range(xc), yv              , expand_range(zc)
    grid_vz = expand_range(xc), expand_range(yc), zv
    
    # Cell fields -------------------------------
    Vx = TA(backend)([vx_stream(x, z) for x in grid_vx[1], y in grid_vx[2], z in grid_vx[3]])
    Vy = TA(backend)([vy_stream(x, z) for x in grid_vy[1], y in grid_vy[2], z in grid_vy[3]])
    Vz = TA(backend)([vz_stream(x, z) for x in grid_vz[1], y in grid_vz[2], z in grid_vz[3]])
    T  = TA(backend)([z for x in xv, y in yv, z in zv])
    T0 = deepcopy(T)
    V  = Vx, Vy, Vz
    dt = min(dx / maximum(abs.(Vx)), dy / maximum(abs.(Vy)), dz / maximum(abs.(Vz))) / 4
    
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 125, 150, 100
    particles = _3D.init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi...
    )
    
    # Advection test
    particle_args = pT, = _3D.init_cell_arrays(particles, Val(1))
    _3D.grid2particle!(pT, xvi, T, particles)
    sumT = sum(T)
   
    niter = 5
    for _ in 1:niter
        _3D.particle2grid!(T, pT, xvi, particles)
        copyto!(T0, T)
        _3D.advection!(particles, _3D.RungeKutta2(), V, (grid_vx, grid_vy, grid_vz), dt)
        _3D.move_particles!(particles, xvi, particle_args)
        # reseed
        _3D.inject_particles!(particles, (pT, ), xvi)
        _3D.grid2particle!(pT, xvi, T, particles)
    end
    sumT_final = sum(T)
    err = abs(sumT - sumT_final) / sumT
    println(err)
    return err
end

function test_advection()
    err = 0e0
    for _ in 1:5
        err = test_advection_3D()
        !isnan(err) && break
    end
    tol = 1e-1
    passed = err < tol
    return passed
end

env = ENV["JULIA_JUSTPIC_BACKEND"]
if env === "AMDGPU" || env === "CUDA"
    @testset "Miniapps" begin
        @test test_advection()
    end
end

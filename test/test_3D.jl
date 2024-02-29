@static if ENV["JULIA_JUSTPIC_BACKEND"] === "AMDGPU"
    using AMDGPU
    AMDGPU.allowscalar(true)
elseif ENV["JULIA_JUSTPIC_BACKEND"] === "CUDA"
    using CUDA
    CUDA.allowscalar(true)
end

using JustPIC, JustPIC._3D, CellArrays, ParallelStencil, Test, LinearAlgebra

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
vx_stream(x, z) =  250 * sin(π*x) * cos(π*z)
vy_stream(x, z) =  0.0
vz_stream(x, z) = -250 * cos(π*x) * sin(π*z)

@testset "Interpolations 3D" begin
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
    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi, dxi, ni
    )
    pT, = init_cell_arrays(particles, Val(1))

    # Linear field at the vertices
    T  = TA(backend)([z for x in xv, y in yv, z in zv])
    T0 = TA(backend)([z for x in xv, y in yv, z in zv])

    # Grid to particle test
    grid2particle!(pT, xvi, T, particles)

    @test pT == particles.coords[3]

    # Grid to particle test
    grid2particle_flip!(pT, xvi, T, T0, particles)

    @test pT == particles.coords[3]

    # Particle to grid test
    T2 = similar(T)
    particle2grid!(T2, pT, xvi, particles)

    @test norm(T2 .- T) / length(T) < 1e-2
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
    particles1 = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi..., dxi..., ni...
    )

    particles2 = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi, dxi, ni
    )

    @test particles1.min_xcell == particles2.min_xcell
    @test particles1.max_xcell == particles2.max_xcell
    @test particles1.np == particles2.np
end

@testset "Cell index 3D" begin
    n = 100
    a, b = rand()*50, rand()*50
    start, finish = extrema((a, b))
    L = finish - start
    x = range(start, stop=finish, length=n)
    xv = x, x, x

    p = px, py, pz = tuple((rand(3).*L .+ start)...)
    i, j, k = cell_index(p, xv)
    @assert x[i] ≤ px < x[i+1]
    @assert x[j] ≤ py < x[j+1]
    @assert x[k] ≤ pz < x[k+1]

    y = x
    z = range(-start, stop=finish, length=n)
    xv = x, y, z
    px, py = tuple((rand(2).*L .+ start)...)
    Lz = z[end] - z[1]
    pz = rand()*Lz - start
    p = px, py, pz

    i, j, k = cell_index(p, xv)
    @assert x[i] ≤ px < x[i+1]
    @assert y[j] ≤ py < y[j+1]
    @assert z[k] ≤ pz < z[k+1]
end

@testset "Passive markers 3D" begin
    n   = 64
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
    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi, dxi, ni
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
        (@rand(np) .+ 1) .* Lx/4
    end

    passive_markers = init_passive_markers(backend, passive_coords);
    T_marker = zeros(np)
    P_marker = zeros(np)

    for _ in 1:75
        advect_passive_markers!(passive_markers, V, grid_vx, grid_vy, grid_vz, dt)
    end

    # interpolate grid fields T and P onto the marker locations
    grid2particle!((T_marker, P_marker), xvi, (T, P), passive_markers)

    x_marker = passive_markers.coords[1]
    z_marker = passive_markers.coords[3]

    @test x_marker ≈ P_marker
    @test z_marker ≈ T_marker
end

function test_advection_3D()
    n   = 64
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
    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi, dxi, ni
    )

    # Cell fields -------------------------------
    Vx = TA(backend)([vx_stream(x, z) for x in grid_vx[1], y in grid_vx[2], z in grid_vx[3]])
    Vy = TA(backend)([vy_stream(x, z) for x in grid_vy[1], y in grid_vy[2], z in grid_vy[3]])
    Vz = TA(backend)([vz_stream(x, z) for x in grid_vz[1], y in grid_vz[2], z in grid_vz[3]])
    T  = TA(backend)([z for x in xv, y in yv, z in zv])
    T0 = deepcopy(T)
    V  = Vx, Vy, Vz

    dt = min(dx / maximum(abs.(Vx)), dy / maximum(abs.(Vy)), dz / maximum(abs.(Vz))) / 2

    # Advection test
    particle_args = pT, = init_cell_arrays(particles, Val(1))
    grid2particle!(pT, xvi, T, particles)

    sumT = sum(T)

    niter = 25
    for _ in 1:niter
        particle2grid!(T, pT, xvi, particles)
        copyto!(T0, T)
        advection_RK!(particles, V, grid_vx, grid_vy, grid_vz, dt, 2 / 3)
        shuffle_particles!(particles, xvi, particle_args)

        # reseed
        inject = check_injection(particles)
        inject && inject_particles!(particles, (pT, ), (T,), xvi)

        grid2particle_flip!(pT, xvi, T, T0, particles)
    end

    sumT_final = sum(T)

    return abs(sumT - sumT_final) / sumT

end

function test_advection()
    err = test_advection_3D()
    tol = 1e-2
    passed = err < tol

    return passed
end

@testset "Miniapps" begin
    @test test_advection()
end
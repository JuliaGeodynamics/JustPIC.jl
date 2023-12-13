using JustPIC, CellArrays, ParallelStencil, Test, LinearAlgebra

@static if occursin("AMDGPU", JustPIC.backend)
    @init_parallel_stencil(AMDGPU, Float64, 2)
    JustPIC.AMDGPU.allowscalar(true)
elseif occursin("CUDA", JustPIC.backend)
    @init_parallel_stencil(CUDA, Float64, 2)
    JustPIC.CUDA.allowscalar(true)
end

function init_particles(nxcell, max_xcell, min_xcell, x, y, dx, dy, nx, ny)
    ni = nx, ny
    ncells = nx * ny
    np = max_xcell * ncells
    px, py = ntuple(_ -> @rand(ni..., celldims=(max_xcell,)) , Val(2))

    inject = @fill(false, nx, ny, eltype=Bool)
    index = @fill(false, ni..., celldims=(max_xcell,), eltype=Bool)

    @parallel_indices (i, j) function fill_coords_index(px, py, index, x, y, dx, dy, nxcell, max_xcell)
        # lower-left corner of the cell
        x0, y0 = x[i], y[j]
        # fill index array
        for l in 1:max_xcell
            if l <= nxcell
                @cell px[l, i, j] = x0 + dx * (@cell(px[l, i, j]) * 0.9 + 0.05)
                @cell py[l, i, j] = y0 + dy * (@cell(py[l, i, j]) * 0.9 + 0.05)
                @cell index[l, i, j] = true

            else
                @cell px[l, i, j] = NaN
                @cell py[l, i, j] = NaN

            end
        end
        return nothing
    end

    @parallel (1:nx, 1:ny) fill_coords_index(px, py, index, x, y, dx, dy, nxcell, max_xcell)

    return Particles(
        (px, py), index, inject, nxcell, max_xcell, min_xcell, np, (nx, ny)
    )
end

function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = round(x1-dx; sigdigits=5)
    xF = round(x2+dx; sigdigits=5)
    range(xI, xF, length=n+2)
end

# Analytical flow solution
vx_stream(x, y) =  250 * sin(π*x) * cos(π*y)
vy_stream(x, y) = -250 * cos(π*x) * sin(π*y)

function advection_test_2D()
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 6, 12, 1
    n = 64
    nx = ny = n-1
    Lx = Ly = 1.0
    # nodal vertices
    xvi = xv, yv = range(0, Lx, length=n), range(0, Ly, length=n)
    dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
    # nodal centers
    xci = xc, yc = range(0+dx/2, Lx-dx/2, length=n-1), range(0+dy/2, Ly-dy/2, length=n-1)
    # staggered grid velocity nodal locations
    grid_vx = xv, expand_range(yc)
    grid_vy = expand_range(xc), yv

    particles = init_particles(
        nxcell, max_xcell, min_xcell, xvi..., dxi..., nx, ny
    )

    # Cell fields -------------------------------
    Vx = TA([vx_stream(x, y) for x in grid_vx[1], y in grid_vx[2]]);
    Vy = TA([vy_stream(x, y) for x in grid_vy[1], y in grid_vy[2]]);
    T  = TA([y for x in xv, y in yv]);
    T0 = deepcopy(T)
    V  = Vx, Vy;

    dt = min(dx / maximum(abs.(Array(Vx))),  dy / maximum(abs.(Array(Vy)))) / 2;

    # Advection test
    particle_args = pT, = init_cell_arrays(particles, Val(1));
    grid2particle!(pT, xvi, T, particles.coords);

    sumT = sum(T)

    niter = 25
    for it in 1:niter
        particle2grid!(T, pT, xvi, particles.coords)
        copyto!(T0, T)
        advection_RK!(particles, V, grid_vx, grid_vy, dt, 2 / 3)
        shuffle_particles!(particles, xvi, particle_args)

        inject = check_injection(particles)
        inject && inject_particles!(particles, (pT, ), (T,), xvi)

        grid2particle_flip!(pT, xvi, T, T0, particles.coords)
    end

    sumT_final = sum(T)

    return abs(sumT - sumT_final) / sumT

end

function test_advection_2D()
    err = advection_test_2D()
    tol = 1e-4
    passed = err < tol

    return passed
end

@testset "Advection 2D" begin
    @test test_advection_2D()
end

# Analytical flow solution
vi_stream(x) =  π*1e-5 * (x - 0.5)

function test_rotating_circle()
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 12, 24, 6
    n = 101
    nx = ny = n-1
    Lx = Ly = 1.0
    # nodal vertices
    xvi = xv, yv = range(0, Lx, length=n), range(0, Ly, length=n)
    dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
    # nodal centers
    xc, yc = range(0+dx/2, Lx-dx/2, length=n-1), range(0+dy/2, Ly-dy/2, length=n-1)
    # staggered grid velocity nodal locations
    grid_vx = xv, expand_range(yc)
    grid_vy = expand_range(xc), yv

    particles = init_particles(
        nxcell, max_xcell, min_xcell, xvi..., dxi..., nx, ny
    )

    # Cell fields -------------------------------
    Vx = TA([-vi_stream(y) for x in grid_vx[1], y in grid_vx[2]]);
    Vy = TA([ vi_stream(x) for x in grid_vy[1], y in grid_vy[2]]);
    xc0 = yc0 =  0.25
    R   = 12 * dx
    T   = TA([((x-xc0)^2 + (y-yc0)^2 ≤ R^2)  * 1.0 for x in xv, y in yv]);
    T0  = deepcopy(T)
    V   = Vx, Vy;

    w      = π * 1e-5  # angular velocity
    period = 1  # revolution number
    tmax   = period / (w/(2*π)) / 10
    dt     = 200.0

    particle_args = pT, = init_cell_arrays(particles, Val(1));
    grid2particle!(pT, xvi, T, particles.coords);

    t = 0
    it = 0
    sumT = sum(T)
    while t ≤ tmax
        particle2grid!(T, pT, xvi, particles.coords)
        copyto!(T0, T)
        advection_RK!(particles, V, grid_vx, grid_vy, dt, 2 / 3)
        shuffle_particles!(particles, xvi, particle_args)

        inject = check_injection(particles)
        inject && inject_particles!(particles, (pT, ), (T,), xvi)

        grid2particle_flip!(pT, xvi, T, T0, particles.coords)

        t += dt
        it += 1
    end

    sumT_final = sum(T)

    return abs(sumT - sumT_final) / sumT
end

function test_advection_2D()
    @show err = test_rotating_circle()
    tol = 5e-3
    passed = err < tol

    return passed
end

@testset "Rotating circle 2D" begin
    @test test_advection_2D()
end

@testset "Interpolations 2D" begin
    nxcell, max_xcell, min_xcell = 24, 24, 1
    n = 5 # number of vertices
    nx = ny = n-1
    Lx = Ly = 1.0
    # nodal vertices
    xvi = xv, yv = range(0, Lx, length=n), range(0, Ly, length=n)
    dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
    # nodal centers
    xci = xc, yc = range(0+dx/2, Lx-dx/2, length=n-1), range(0+dy/2, Ly-dy/2, length=n-1)
    # staggered grid velocity nodal locations

    # Initialize particles & particle fields
    particles = init_particles(
        nxcell, max_xcell, min_xcell, xvi..., dxi..., nx, ny
    )
    pT, = init_cell_arrays(particles, Val(1))

    # Linear field at the vertices
    T  = TA([y for x in xv, y in yv])
    T0 = TA([y for x in xv, y in yv])

    # Grid to particle test
    grid2particle!(pT, xvi, T, particles.coords)

    @test pT == particles.coords[2]

    # Grid to particle test
    grid2particle_flip!(pT, xvi, T, T0, particles.coords)

    @test pT == particles.coords[2]

    # Particle to grid test
    T2 = similar(T)
    particle2grid!(T2, pT, xvi, particles.coords)

    @test norm(T2 .- T) / length(T) < 1e-2
end

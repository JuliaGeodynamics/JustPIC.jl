using JustPIC, CellArrays, ParallelStencil, Test, LinearAlgebra
import JustPIC: @idx, @cell

@static if occursin("AMDGPU", JustPIC.backend)
    @init_parallel_stencil(AMDGPU, Float64, 3)
    JustPIC.AMDGPU.allowscalar(true)
elseif occursin("CUDA", JustPIC.backend)
    @init_parallel_stencil(CUDA, Float64, 3)
    JustPIC.CUDA.allowscalar(true)
end

function init_particles(nxcell, max_xcell, min_xcell, x, y, z, dx, dy, dz, ni)
    ncells     = prod(ni)
    np         = max_xcell * ncells
    px, py, pz = ntuple(_ -> @fill(NaN, ni..., celldims=(max_xcell,)) , Val(3))
    inject     = @fill(false, ni..., eltype=Bool)
    index      = @fill(false, ni..., celldims=(max_xcell,), eltype=Bool)

    @parallel_indices (i, j, k) function fill_coords_index(px, py, pz, index)
        # lower-left corner of the cell
        x0, y0, z0 = x[i], y[j], z[k]
        # fill index array
        for l in 1:nxcell
            @cell px[l, i, j, k]    = x0 + dx * rand(0.05:1e-5:0.95)
            @cell py[l, i, j, k]    = y0 + dy * rand(0.05:1e-5:0.95)
            @cell pz[l, i, j, k]    = z0 + dz * rand(0.05:1e-5:0.95)
            @cell index[l, i, j, k] = true
        end
        return nothing
    end

    @parallel (@idx ni) fill_coords_index(px, py, pz, index)

    return Particles(
        (px, py, pz), index, inject, nxcell, max_xcell, min_xcell, np, ni
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
vx_stream(x, z) =  250 * sin(π*x) * cos(π*z)
vy_stream(x, z) =  0.0
vz_stream(x, z) = -250 * cos(π*x) * sin(π*z)

function test_advection_3D()
    n   = 64
    nx  = ny = nz = n-1
    Lx  = Ly = Lz = 1.0
    ni  = nx, ny, nz
    Li  = Lx, Ly, Lz
    # nodal vertices
    xvi = xv, yv, zv = ntuple(i -> range(0, Li[i], length=n), Val(3))
    # grid spacing
    dxi = dx, dy, dz = ntuple(i -> xvi[i][2] - xvi[i][1], Val(3))
    # nodal centers
    xci = xc, yc, zc = ntuple(i -> range(0+dxi[i]/2, Li[i]-dxi[i]/2, length=ni[i]), Val(3))

    # staggered grid velocity nodal locations
    grid_vx = xv              , expand_range(yc), expand_range(zc)
    grid_vy = expand_range(xc), yv              , expand_range(zc)
    grid_vz = expand_range(xc), expand_range(yc), zv

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 24, 24, 3
    particles = init_particles(
        nxcell, max_xcell, min_xcell, xvi..., dxi..., ni
    )

    # Cell fields -------------------------------
    Vx = TA([vx_stream(x, z) for x in grid_vx[1], y in grid_vx[2], z in grid_vx[3]])
    Vy = TA([vy_stream(x, z) for x in grid_vy[1], y in grid_vy[2], z in grid_vy[3]])
    Vz = TA([vz_stream(x, z) for x in grid_vz[1], y in grid_vz[2], z in grid_vz[3]])
    T  = TA([z for x in xv, y in yv, z in zv])
    T0 = deepcopy(T)
    V  = Vx, Vy, Vz

    dt = min(dx / maximum(abs.(Vx)), dy / maximum(abs.(Vy)), dz / maximum(abs.(Vz))) / 2

    # Advection test
    particle_args = pT, = init_cell_arrays(particles, Val(1))
    grid2particle!(pT, xvi, T, particles.coords)

    sumT = sum(T)

    niter = 100
    for _ in 1:niter
        particle2grid!(T, pT, xvi, particles.coords)
        copyto!(T0, T)
        advection_RK!(particles, V, grid_vx, grid_vy, grid_vz, dt, 2 / 3)
        shuffle_particles!(particles, xvi, particle_args)

        # reseed
        inject = check_injection(particles)
        inject && inject_particles!(particles, (pT, ), (T,), xvi)

        grid2particle_flip!(pT, xvi, T, T0, particles.coords)
    end

    sumT_final = sum(T)

    return abs(sumT - sumT_final) / sumT

end

function test_advection()
    err = test_advection_3D()
    tol = 1e-3
    passed = err < tol

    return passed
end

@testset begin
    @test test_advection()
end

@testset "Interpolations 3D" begin
    nxcell, max_xcell, min_xcell = 24, 24, 1
    n   = 5 # number of vertices
    nx  = ny = nz = n-1
    ni  = nx, ny, nz
    Lx  = Ly = Lz = 1.0
    Li  = Lx, Ly, Lz
    # nodal vertices
    xvi = xv, yv, zv = ntuple(i -> range(0, Li[i], length=n), Val(3))
    # grid spacing
    dxi = dx, dy, dz = ntuple(i -> xvi[i][2] - xvi[i][1], Val(3))
    # nodal centers
    xci = xc, yc, zc = ntuple(i -> range(0+dxi[i]/2, Li[i]-dxi[i]/2, length=ni[i]), Val(3))

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 24, 24, 1
    particles = init_particles(
        nxcell, max_xcell, min_xcell, xvi..., dxi..., ni
    )
    pT, = init_cell_arrays(particles, Val(1))

    # Linear field at the vertices
    T  = TA([z for x in xv, y in yv, z in zv])
    T0 = TA([z for x in xv, y in yv, z in zv])

    # Grid to particle test
    grid2particle!(pT, xvi, T, particles.coords)

    @test pT == particles.coords[3]

    # Grid to particle test
    grid2particle_flip!(pT, xvi, T, T0, particles.coords)

    @test pT == particles.coords[3]

    # Particle to grid test
    T2 = similar(T)
    particle2grid!(T2, pT, xvi, particles.coords)

    @test norm(T2 .- T) / length(T) < 1e-2
end

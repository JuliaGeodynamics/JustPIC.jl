using JustPIC, JustPIC._3D

# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA"),
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU")
const backend = JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

using ParallelStencil
@init_parallel_stencil(Threads, Float64, 3)

using GLMakie
using ImplicitGlobalGrid
using MPI: MPI

# Analytical flow solution
vx_stream(x, z) = 250 * sin(π * x) * cos(π * z)
vy_stream(x, z) = 0.0
vz_stream(x, z) = -250 * cos(π * x) * sin(π * z)
g(x) = Point2f(
    vx_stream(x[1], x[3]),
    vy_stream(x[1], x[3]),
    vz_stream(x[1], x[3]),
)

function main()
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 25, 40, 15
    n = 32
    nx = ny = nz = n - 1
    me, dims, = init_global_grid(n - 1, n - 1, n - 1; init_MPI = MPI.Initialized() ? false : true)
    Lx = Ly = Lz = 1.0
    dxi = dx, dy, dz = Lx / (nx_g() - 1), Ly / (ny_g() - 1), Lz / (nz_g() - 1)
    # nodal vertices
    xvi = xv, yv, zv = let
        dummy = zeros(n, n, n)
        xv = TA(backend)([x_g(i, dx, dummy) for i in axes(dummy, 1)])
        yv = TA(backend)([y_g(i, dy, dummy) for i in axes(dummy, 2)])
        zv = TA(backend)([z_g(i, dz, dummy) for i in axes(dummy, 3)])
        xv, yv, zv
    end
    # nodal centers
    xci = xc, yc, zc = let
        dummy = zeros(nx, ny, nz)
        xc = TA(backend)([x_g(i, dx, dummy) for i in axes(dummy, 1)])
        yc = TA(backend)([y_g(i, dy, dummy) for i in axes(dummy, 2)])
        zc = TA(backend)([z_g(i, dz, dummy) for i in axes(dummy, 3)])
        xc, yc, zc
    end

    # staggered grid for the velocity components
    grid_vx = xv, add_ghost_nodes(yc, dy, (0.0, Ly)), add_ghost_nodes(zc, dz, (0.0, Lz))
    grid_vy = add_ghost_nodes(xc, dx, (0.0, Lx)), yv, add_ghost_nodes(zc, dz, (0.0, Lz))
    grid_vz = add_ghost_nodes(xc, dx, (0.0, Lx)), add_ghost_nodes(yc, dy, (0.0, Ly)), zv

    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi...
    )

    # Cell fields -------------------------------
    Vx = TA(backend)([vx_stream(x, z) for x in grid_vx[1], y in grid_vx[2], z in grid_vx[3]])
    Vy = TA(backend)([vy_stream(x, z) for x in grid_vy[1], y in grid_vy[2], z in grid_vy[3]])
    Vz = TA(backend)([vz_stream(x, z) for x in grid_vz[1], y in grid_vz[2], z in grid_vz[3]])
    T = TA(backend)([z for x in xv, y in yv, z in zv])
    V = Vx, Vy, Vz

    # plotting buffers
    nx_v = (size(T, 1) - 2) * dims[1]
    ny_v = (size(T, 2) - 2) * dims[2]
    nz_v = (size(T, 3) - 2) * dims[3]
    T_v = zeros(nx_v, ny_v, nz_v)
    T_nohalo = @zeros(size(T) .- 2)

    dt = mapreduce(x -> x[1] / MPI.Allreduce(maximum(abs.(x[2])), MPI.MAX, MPI.COMM_WORLD), min, zip(dxi, V))

    # Advection test
    particle_args = pT, = init_cell_arrays(particles, Val(1))
    grid2particle!(pT, xvi, T, particles)

    niter = 125 #250
    for iter in 1:niter
        me == 0 && @show iter

        # advect particles
        advection!(particles, RungeKutta2(), V, (grid_vx, grid_vy, grid_vz), dt)

        # update halos
        update_cell_halo!(particles.coords...)
        update_cell_halo!(particle_args...)
        update_cell_halo!(particles.index)

        # shuffle particles
        move_particles!(particles, xvi, particle_args)
        # interpolate T from particle to grid
        particle2grid!(T, pT, xvi, particles)

        @views T_nohalo .= T[2:(end - 1), 2:(end - 1), 2:(end - 1)]
        gather!(T_nohalo, T_v)

        if me == 0 && iter % 10 == 0
            x_global = range(0, Lx, length = size(T_v, 1))
            z_global = range(0, Lz, length = size(T_v, 3))
            f, = heatmap(x_global, z_global, T_v[:, 2, :])
            save("figs/T_MPI_$iter.png", f)
        end

    end

    return finalize_global_grid()

end

main()

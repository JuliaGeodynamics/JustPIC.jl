using JustPIC

# Backend is selected via the JULIA_JUSTPIC_BACKEND environment variable.
# Options: "CPU" (default), "CUDA", "AMDGPU"
const backend_name = get(ENV, "JULIA_JUSTPIC_BACKEND", "CPU")
const backend = if backend_name == "CUDA"
    using CUDA
    CUDA.CUDABackend
elseif backend_name == "AMDGPU"
    using AMDGPU
    AMDGPU.ROCBackend
elseif backend_name == "CPU"
    JustPIC.CPU
else
    error("Unknown backend: $backend_name. Options: \"CPU\", \"CUDA\", \"AMDGPU\"")
end

# using GLMakie
using ImplicitGlobalGrid
import MPI

# Analytical flow solution
vx_stream(x, y) = 250 * sin(π * x) * cos(π * y)
vy_stream(x, y) = -250 * cos(π * x) * sin(π * y)
g(x) = Point2f(
    vx_stream(x[1], x[2]),
    vy_stream(x[1], x[2])
)

function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = x1 - dx
    xF = x2 + dx
    return LinRange(xI, xF, n + 2)
end

function main()
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 24, 40, 1
    n = parse(Int, get(ENV, "JUSTPIC_MPI_CI_N", "64"))
    nx = ny = n - 1
    me, dims, = init_global_grid(
        n - 1, n - 1, 1;
        init_MPI = MPI.Initialized() ? false : true,
        select_device = false,
        periodx = 1,
        periody = 1,
    )
    Lx = Ly = 1.0
    dxi = dx, dy = Lx / (nx_g() - 1), Ly / (ny_g() - 1)
    # Local grids are anchored at the first interior point: with periodic boundaries,
    # x_g/y_g wrap the halo coordinates into [0, L), so the endpoints of a boundary
    # rank do not lie on one contiguous local grid.
    local_range(x2, d, len) = LinRange(x2 - d, x2 + (len - 2) * d, len)
    # nodal vertices
    xvi = xv, yv = let
        dummy = zeros(n, n)
        local_range(x_g(2, dx, dummy), dx, n), local_range(y_g(2, dy, dummy), dy, n)
    end
    # nodal centers
    xci = xc, yc = let
        dummy = zeros(nx, ny)
        local_range(x_g(2, dx, dummy), dx, nx), local_range(y_g(2, dy, dummy), dy, ny)
    end

    # staggered grid for the velocity components
    grid_vx = xv, expand_range(yc)
    grid_vy = expand_range(xc), yv

    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, grid_vx, grid_vy
    )

    # Cell fields -------------------------------
    Vx = TA(backend)([vx_stream(x, y) for x in grid_vx[1], y in grid_vx[2]])
    Vy = TA(backend)([vy_stream(x, y) for x in grid_vy[1], y in grid_vy[2]])
    xvi_particles = Array.(particles.xvi)
    T = TA(backend)([y for x in xvi_particles[1], y in xvi_particles[2]])
    T0 = deepcopy(T)
    V = Vx, Vy

    nx_v = (size(T, 1) - 2) * dims[1]
    ny_v = (size(T, 2) - 2) * dims[2]
    T_v = zeros(nx_v, ny_v)
    T_nohalo = TA(backend)(zeros(size(T) .- 2))

    dt = mapreduce(x -> x[1] / MPI.Allreduce(maximum(abs.(x[2])), MPI.MAX, MPI.COMM_WORLD), min, zip(dxi, V)) / 2

    # Advection test
    particle_args = pT, = init_cell_arrays(particles, Val(1))
    grid2particle!(pT, T, particles)

    niter = parse(Int, get(ENV, "JUSTPIC_MPI_CI_NITER", "250"))
    for iter in 1:niter
        me == 0 && @show iter

        # advect particles
        advection!(particles, RungeKutta2(), V, dt)

        # update halos
        update_cell_halo!(particles.coords..., particle_args..., particles.index)
        # shuffle particles
        move_particles!(particles, particle_args)
        inject_particles!(particles, ())
        grid2particle!(pT, T, particles)
        update_cell_halo!(particles.coords..., particle_args..., particles.index)
        # interpolate T from particle to grid
        particle2grid!(T, pT, particles)

        @views T_nohalo .= T[2:(end - 1), 2:(end - 1)]
        all(isfinite, Array(T_nohalo)) || error("Non-finite reconstructed field on rank $me")
        gather!(Array(T_nohalo), T_v)
    end
    return finalize_global_grid()

end

main()

using JustPIC
using JustPIC._3D

# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA"),
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU")
const backend = JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

using GLMakie

function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = x1 - dx
    xF = x2 + dx
    return LinRange(xI, xF, n + 2)
end

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

    # Cell fields -------------------------------
    Vx = TA(backend)([vx_stream(x, z) for x in grid_vx[1], y in grid_vx[2], z in grid_vx[3]])
    Vy = TA(backend)([vy_stream(x, z) for x in grid_vy[1], y in grid_vy[2], z in grid_vy[3]])
    Vz = TA(backend)([vz_stream(x, z) for x in grid_vz[1], y in grid_vz[2], z in grid_vz[3]])
    T = TA(backend)([z for x in xv, y in yv, z in zv])
    T0 = TA(backend)([z for x in xv, y in yv, z in zv])
    V = Vx, Vy, Vz

    dt = min(dx / maximum(abs.(Vx)), dy / maximum(abs.(Vy)), dz / maximum(abs.(Vz))) / 2
    dt *= 0.75

    !isdir("figs") && mkdir("figs")

    niter = 250
    for it in 1:niter
        semilagrangian_advection!(T, T0, RungeKutta2(), V, (grid_vx, grid_vy, grid_vz), xvi, dt)
        T[:, 1, :] .= T[:, 2, :]
        T[:, end, :] .= T[:, end - 1, :]
        T[1, :, :] .= T[2, :, :]
        T[end, :, :] .= T[end - 1, :, :]

        copyto!(T0, T)

        if rem(it, 10) == 0
            f, ax, = heatmap(xvi[1:2]..., Array(T[:, 1, :]), colormap = :batlow)
            save("figs/test_$(it).png", f)
            f
        end
    end

    return println("Finished")
end

main()

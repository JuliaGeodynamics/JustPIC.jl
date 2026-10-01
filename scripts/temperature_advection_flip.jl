using JustPIC

# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA"),
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU")
const backend = JustPIC.CPU # Options: JustPIC.CPU, CUDA.CUDABackend, AMDGPU.ROCBackend

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
vx_stream(x, y) = sin(π * x) * cos(π * y)
vy_stream(x, y) = -cos(π * x) * sin(π * y)

# Explicit diffusion step on a centroid field with a ghost layer (zero-flux boundaries)
function diffuse!(T, κ, dt, dx, dy)
    Ti = @view T[2:(end - 1), 2:(end - 1)]
    ΔT = @views @. (T[3:end, 2:(end - 1)] - 2 * Ti + T[1:(end - 2), 2:(end - 1)]) / dx^2 +
        (T[2:(end - 1), 3:end] - 2 * Ti + T[2:(end - 1), 1:(end - 2)]) / dy^2
    Ti .+= κ .* dt .* ΔT
    @views begin
        T[1, :] .= T[2, :]
        T[end, :] .= T[end - 1, :]
        T[:, 1] .= T[:, 2]
        T[:, end] .= T[:, end - 1]
    end
    return nothing
end

function main(; α = 0.1, κ = 1.0e-3, niter = 100)
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 24, 30, 12
    n = 64
    Lx = Ly = 1.0
    # nodal vertices
    xvi = xv, yv = LinRange(0, Lx, n), LinRange(0, Ly, n)
    dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
    # nodal centers
    xci = xc, yc = LinRange(0 + dx / 2, Lx - dx / 2, n - 1), LinRange(0 + dy / 2, Ly - dy / 2, n - 1)
    # staggered grid velocity nodal locations
    grid_vx = xv, expand_range(yc)
    grid_vy = expand_range(xc), yv

    particles = init_particles(backend, nxcell, max_xcell, min_xcell, grid_vx, grid_vy)

    # Grid fields: velocity, and temperature at the centroids and vertices (with ghost nodes)
    Vx = TA(backend)([vx_stream(x, y) for x in grid_vx[1], y in grid_vx[2]])
    Vy = TA(backend)([vy_stream(x, y) for x in grid_vy[1], y in grid_vy[2]])
    V = Vx, Vy
    xci_p, xvi_p = Array.(particles.xci), Array.(particles.xvi)
    blob(x, y) = exp(-((x - 0.5)^2 + (y - 0.7)^2) / 0.01)
    T = TA(backend)([blob(x, y) for x in xci_p[1], y in xci_p[2]]) # centroids
    Tv = TA(backend)([blob(x, y) for x in xvi_p[1], y in xvi_p[2]]) # vertices
    T0 = similar(T)

    dt = 0.5 * min(dx / maximum(abs.(Array(Vx))), dy / maximum(abs.(Array(Vy))))
    @assert κ * dt * (1 / dx^2 + 1 / dy^2) < 0.5 "explicit diffusion is unstable; reduce κ or dt"

    # Particle fields: temperature and its value before the grid update
    particle_args = pT, pT0 = init_cell_arrays(particles, Val(2))
    centroid2particle!(pT, T, particles)

    !isdir("figs") && mkdir("figs")

    for it in 1:niter
        advection!(particles, RungeKutta2(), V, dt)
        move_particles!(particles, particle_args)
        inject_particles!(particles, (pT,))

        # PIC: particles → centroids, which are then diffused on the grid
        particle2centroid!(T, pT, particles)
        T0 .= T
        diffuse!(T, κ, dt, dx, dy)

        # FLIP: particles keep their own temperature and receive only the grid increment,
        # blended with the PIC value by α (α = 1 is pure PIC, α = 0 pure FLIP)
        pT0.data .= pT.data
        centroid2particle_flip!(pT, T, T0, particles; α)

        # The vertex temperature follows the same particle increment, so the vertex
        # field never goes through a PIC average of the particles themselves
        particle2grid_flip!(Tv, pT, pT0, particles)

        if rem(it, 10) == 0
            @show it, extrema(T)
            f = Figure(size = (1000, 450))
            ax1 = Axis(f[1, 1], title = "centroids (PIC → diffusion)", aspect = 1)
            ax2 = Axis(f[1, 2], title = "vertices (particle2grid_flip!)", aspect = 1)
            heatmap!(ax1, xci..., Array(T)[2:(end - 1), 2:(end - 1)], colormap = :batlow)
            heatmap!(ax2, xvi..., Array(Tv)[2:(end - 1), 2:(end - 1)], colormap = :batlow)
            save("figs/flip_$(it).png", f)
        end
    end

    return println("Finished")
end

main()

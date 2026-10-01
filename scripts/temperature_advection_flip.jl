using JustPIC
using LinearAlgebra

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

# Zero-flux boundaries: copy the first physical centroid into the ghost layer
function neumann!(T)
    @views begin
        T[1, :] .= T[2, :]
        T[end, :] .= T[end - 1, :]
        T[:, 1] .= T[:, 2]
        T[:, end] .= T[:, end - 1]
    end
    return nothing
end

# Explicit diffusion step on a centroid field with a ghost layer
function diffuse!(T, κ, dt, dx, dy)
    Ti = @view T[2:(end - 1), 2:(end - 1)]
    ΔT = @views @. (T[3:end, 2:(end - 1)] - 2 * Ti + T[1:(end - 2), 2:(end - 1)]) / dx^2 +
        (T[2:(end - 1), 3:end] - 2 * Ti + T[2:(end - 1), 1:(end - 2)]) / dy^2
    Ti .+= κ .* dt .* ΔT
    neumann!(T)
    return nothing
end

function main(; α = 0.0, κ = 1.0e-3, niter = 100)
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

    # Grid fields: velocity and the temperature at the centroids (with ghost nodes)
    Vx = TA(backend)([vx_stream(x, y) for x in grid_vx[1], y in grid_vx[2]])
    Vy = TA(backend)([vy_stream(x, y) for x in grid_vy[1], y in grid_vy[2]])
    V = Vx, Vy
    xci_p = Array.(particles.xci)
    blob(x, y) = exp(-((x - 0.5)^2 + (y - 0.7)^2) / 0.01)
    T = TA(backend)([blob(x, y) for x in xci_p[1], y in xci_p[2]]) # temperature stored on the grid
    Tg = similar(T) # temperature after the grid-side (diffusion) update

    dt = 0.5 * min(dx / maximum(abs.(Array(Vx))), dy / maximum(abs.(Array(Vy))))
    @assert κ * dt * (1 / dx^2 + 1 / dy^2) < 0.5 "explicit diffusion is unstable; reduce κ or dt"

    # Particle fields: temperature and its value before the update
    particle_args = pT, pT0 = init_cell_arrays(particles, Val(2))
    centroid2particle!(pT, T, particles)

    !isdir("figs") && mkdir("figs")

    for it in 1:niter
        advection!(particles, RungeKutta2(), V, dt)
        move_particles!(particles, particle_args)
        inject_particles!(particles, (pT,))
        particle2centroid!(T, pT, particles)
        # particle2centroid! fills only the interior; the particle-side increment below
        # reads the ghost centroids, which must match those of the diffused copy
        neumann!(T)

        # 1. Grid-side update: diffuse a copy of T
        Tg .= T
        diffuse!(Tg, κ, dt, dx, dy)

        # 2. Grid → particles: interpolate the increment Tg - T and add it to the particles
        #    (α = 0 is pure FLIP: pT += interp(Tg - T))
        pT0.data .= pT.data
        centroid2particle_flip!(pT, Tg, T, particles; α)

        # 3. Particles → grid: interpolate the particle increment pT - pT0 and add it to T,
        #    T += Σω (pT - pT0) / Σω
        particle2centroid_flip!(T, pT, pT0, particles)

        if rem(it, 10) == 0
            err = norm(Array(T .- Tg)[2:(end - 1), 2:(end - 1)]) / norm(Array(Tg)[2:(end - 1), 2:(end - 1)])
            @show it, extrema(T), err
            f = Figure(size = (1100, 450))
            ax1 = Axis(f[1, 1], title = "T after the particle round trip", aspect = 1)
            ax2 = Axis(f[1, 3], title = "T - Tg (round trip vs grid update)", aspect = 1)
            hm1 = heatmap!(ax1, xci..., Array(T)[2:(end - 1), 2:(end - 1)], colormap = :batlow)
            hm2 = heatmap!(ax2, xci..., log10.(Array(T .- Tg)[2:(end - 1), 2:(end - 1)]), colormap = :vik)
            Colorbar(f[1, 2], hm1, label = "T")
            Colorbar(f[1, 4], hm2, label = "T - Tg")
            save("figs/flip_$(it).png", f)
        end
    end

    return println("Finished")
end

main(; α = 0)

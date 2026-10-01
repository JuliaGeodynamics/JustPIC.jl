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

# Grid-side update of T followed by the particle round trip:
# T += Σω (pT - pT0) / Σω, where pT has received the interpolated increment Tg - T.
# α = 0 is pure FLIP, α = 1 pure PIC.
function round_trip!(T, Tg, pT, pT0, particles, α, κ, dt, dx, dy)
    particle2centroid!(T, pT, particles)
    # particle2centroid! fills only the interior; the particle-side increment below
    # reads the ghost centroids, which must match those of the diffused copy
    neumann!(T)

    # 1. Grid-side update: diffuse a copy of T
    Tg .= T
    diffuse!(Tg, κ, dt, dx, dy)

    # 2. Grid → particles: pT += interp(Tg - T), blended with the PIC value by α
    pT0.data .= pT.data
    centroid2particle_flip!(pT, Tg, T, particles; α)

    # 3. Particles → grid: add the interpolated particle increment pT - pT0 to T
    particle2centroid_flip!(T, pT, pT0, particles)
    return nothing
end

interior(A) = Array(A)[2:(end - 1), 2:(end - 1)]
relative_difference(T, Tg) = norm(interior(T .- Tg)) / norm(interior(Tg))

function main(; α_flip = 0.0, α_pic = 1.0, κ = 1.0e-3, niter = 100)
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

    # Grid fields: velocity and two copies of the temperature at the centroids (with ghost nodes),
    # one updated with FLIP and one with PIC
    Vx = TA(backend)([vx_stream(x, y) for x in grid_vx[1], y in grid_vx[2]])
    Vy = TA(backend)([vy_stream(x, y) for x in grid_vy[1], y in grid_vy[2]])
    V = Vx, Vy
    xci_p = Array.(particles.xci)
    blob(x, y) = exp(-((x - 0.5)^2 + (y - 0.7)^2) / 0.01)
    T_flip = TA(backend)([blob(x, y) for x in xci_p[1], y in xci_p[2]])
    T_pic = copy(T_flip)
    Tg_flip, Tg_pic = similar(T_flip), similar(T_flip) # after the grid-side (diffusion) update

    dt = 0.5 * min(dx / maximum(abs.(Array(Vx))), dy / maximum(abs.(Array(Vy))))
    @assert κ * dt * (1 / dx^2 + 1 / dy^2) < 0.5 "explicit diffusion is unstable; reduce κ or dt"

    # Particle fields: temperature and its value before the update, for each copy
    particle_args = pT_flip, pT0_flip, pT_pic, pT0_pic = init_cell_arrays(particles, Val(4))
    centroid2particle!(pT_flip, T_flip, particles)
    centroid2particle!(pT_pic, T_pic, particles)

    !isdir("figs") && mkdir("figs")

    for it in 1:niter
        advection!(particles, RungeKutta2(), V, dt)
        move_particles!(particles, particle_args)
        inject_particles!(particles, (pT_flip, pT_pic))

        round_trip!(T_flip, Tg_flip, pT_flip, pT0_flip, particles, α_flip, κ, dt, dx, dy)
        round_trip!(T_pic, Tg_pic, pT_pic, pT0_pic, particles, α_pic, κ, dt, dx, dy)

        if rem(it, 10) == 0
            @show it, extrema(T_flip), relative_difference(T_flip, Tg_flip)
            @show it, extrema(T_pic), relative_difference(T_pic, Tg_pic)

            # same color scales in both rows
            Tmax = max(maximum(interior(T_flip)), maximum(interior(T_pic)))
            dmax = max(maximum(abs, interior(T_flip .- Tg_flip)), maximum(abs, interior(T_pic .- Tg_pic)))
            f = Figure(size = (1100, 900))
            for (row, name, α, T, Tg) in (
                    (1, "FLIP", α_flip, T_flip, Tg_flip), (2, "PIC", α_pic, T_pic, Tg_pic),
                )
                ax1 = Axis(f[row, 1], title = "$name (α = $α): T after the round trip", aspect = 1)
                ax2 = Axis(f[row, 3], title = "$name (α = $α): T - Tg", aspect = 1)
                hm1 = heatmap!(ax1, xci..., interior(T), colormap = :batlow, colorrange = (0, Tmax))
                hm2 = heatmap!(ax2, xci..., interior(T .- Tg), colormap = :vik, colorrange = (-dmax, dmax))
                Colorbar(f[row, 2], hm1, label = "T")
                Colorbar(f[row, 4], hm2, label = "T - Tg")
            end
            save("figs/flip_$(it).png", f)
        end
    end

    return println("Finished")
end

main()

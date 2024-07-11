using DelimitedFiles
using GLMakie

using JustPIC
using JustPIC._2D
const backend = JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

# Threads is the default backend, 
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA"), 
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU")
const backend = JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend


function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = round(x1-dx; sigdigits=5)
    xF = round(x2+dx; sigdigits=5)
    range(xI, xF, length=n+2)
end

# read velocity from Taras
Vx0 = Array(readdlm("Taras/Vx.txt", '\t')')
Vy0 = Array(readdlm("Taras/Vy.txt", '\t')')

Vx = @. (Vx0[1:end-1, :] + Vx0[2:end, :])/2
Vy = @. (Vy0[:, 1:end-1] + Vy0[:, 2:end])/2
V  = Vx, Vy

# function main(V, m, inject)
    lx   = 100 # Horizontal model size, m
    ly   = 100 # Vertical model size, m
    nx   = 40  # Horizontal grid resolution
    ny   = 40  # Vertical grid resolution
    nx_v = nx+1
    ny_v = ny+1
    dx   = lx/(nx) # Horizontal grid step, m
    dy   = ly/(ny) # Vertical grid step, m
    x    = range( 0, lx, length = nx_v)  # Horizontal coordinates of basic grid points, m
    y    = range( 0, ly, length = ny_v)  # Vertical coordinates of basic grid points, m

    # nodal centers
    xc, yc = range(0+dx/2, lx-dx/2, length=nx), range(0+dy/2, ly-dy/2, length=ny)
    # staggered grid velocity nodal locations
    grid_vx = x, expand_range(yc)
    grid_vy = expand_range(xc), y

    xvi = x, y

    grid_vxi = (
        grid_vx,
        grid_vy,
    )

    nxcell, max_xcell, min_xcell = 4, 50, 4
    # nodal vertices
    xvi = x, y 

    dt = 2.2477e7
    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi...,
    )
    advection!(particles, RungeKutta2(), V, grid_vxi, dt)

    # particle_args = pT, = init_cell_arrays(particles, Val(1));

    niter   = 1
    m       = 3
    inject  = false
    # ntime = 1000000
    ntime   = 10000
    @show inject
    np = Int64[]
    for it in 1:ntime
        if m == 1
            advection!(particles, RungeKutta2(), V, grid_vxi, dt)
        elseif m == 2
            advection_LinP!(particles, RungeKutta2(), V, grid_vxi, dt)
        elseif m == 3
            advection_MQS!(particles, RungeKutta2(), V, grid_vxi, dt)
        end
        move_particles!(particles, xvi, ())
        inject && inject_particles!(particles, (), xvi)

        push!(np, sum(particles.index.data))
    end
#     return particles, np
# end

pxx, pyy  = particles.coords
scatter( pxx.data[:], pyy.data[:], markersize = 4)

1

# # read velocity from Taras
# Vx0 = Array(readdlm("Taras/Vx.txt", '\t')')
# Vy0 = Array(readdlm("Taras/Vy.txt", '\t')')

# Vx = @. (Vx0[1:end-1, :] + Vx0[2:end, :])/2
# Vy = @. (Vy0[:, 1:end-1] + Vy0[:, 2:end])/2
# V  = Vx, Vy

# particles = []
# np        = []
# for inject in (false, ), m in (1,2,3)
#     p, n = main(V, m, inject)
#     push!(particles, p)
#     push!(np, n)
# end

# # titles = ["Linear", "LinP", "MQS"]
# # fig    = Figure(size = (1200, 800))
# # ax     = Axis(fig[1, 1])
# # for i in 4:6
# #     npi = np[i]
# #     lines!(ax, 1:length(npi), (npi)./(20*40^2), label=titles[i-3])
# # end
# # ax.ylabel = "capacity"
# # ax.xlabel = "iteration"
# # axislegend(ax, position =  :lt)
# # fig


# titles = ["Linear", "LinP", "MQS"]
# fig    = Figure(size = (1200, 800))
# ax1    = [Axis(fig[1, i], aspect = 1, title=titles[i]) for i in 1:3]
# # ax2    = [Axis(fig[2, i], aspect = 1) for i in 1:3]
# for i in 1:3
#     # no injection
#     pxx, pyy  = particles[i].coords
#     scatter!(ax1[i], pxx.data[:], pyy.data[:], markersize = 4)
#     hidexdecorations!(ax1[i], label = true, ticklabels = true, ticks = true)
#     if i > 1
#         hideydecorations!(ax1[i], label = true, ticklabels = true, ticks = true)
#     end
#     xlims!(ax1[i], 0, 100)
#     ylims!(ax1[i], 0, 100)
#     # # injection
#     # pxx, pyy  = particles[i+3].coords
#     # scatter!(ax2[i], pxx.data[:], pyy.data[:], markersize = 4)
#     # if i > 1
#     #     hideydecorations!(ax2[i], label = true, ticklabels = true, ticks = true)
#     # end
#     # xlims!(ax2[i], 0, 100)
#     # ylims!(ax2[i], 0, 100)
# end
# fig

# # poly!(
# #     ax1[1],
# #     Rect(0, 0, 25, 25),
# #     color = :red,
# #     alpha = 0.2,
# #     strokewidth = 3,
# # )

# # titles = ["Linear", "LinP", "MQS"]
# # fig    = Figure(size = (1200, 800))
# # ax1    = [Axis(fig[1, i], aspect = 1, title=titles[i]) for i in 1:3]
# # ax2    = [Axis(fig[2, i], aspect = 1) for i in 1:3]
# # for i in 1:3
# #     # no injection
# #     pxx, pyy  = particles[i].coords
# #     scatter!(ax1[i], pxx.data[:], pyy.data[:], markersize = 4)
# #     hidexdecorations!(ax1[i], label = true, ticklabels = true, ticks = true)
# #     if i > 1
# #         hideydecorations!(ax1[i], label = true, ticklabels = true, ticks = true)
# #     end
# #     hidedecorations!(ax1[i], label = false, ticklabels = false, ticks = true)
# #     xlims!(ax1[i], 0, 25)
# #     ylims!(ax1[i], 75, 100)

# #     # injection
# #     pxx, pyy  = particles[i+3].coords
# #     scatter!(ax2[i], pxx.data[:], pyy.data[:], markersize = 4)
# #     if i > 1
# #         hideydecorations!(ax2[i], label = true, ticklabels = true, ticks = true)
# #     end
# #     hidedecorations!(ax2[i], label = false, ticklabels = false, ticks = true)
# #     xlims!(ax2[i], 0, 50)
# #     ylims!(ax2[i], 50, 100)
# # end
# # fig


# # c = [[sum(I) for I in particles[1].index] for p in particles[4:6]]

# # f, ax, h = heatmap(c[1], colormap = :oleron, colorrange = (0, 20))
# # Colorbar(f[1,2], h)
# # f

# # f, ax, h = heatmap(c[2], colormap = :oleron, colorrange = (0, 20))
# # Colorbar(f[1,2], h)
# # f

# # f, ax, h = heatmap(c[3], colormap = :oleron, colorrange = (0, 20))
# # Colorbar(f[1,2], h)
# # f

# # extrema(c)

# # findmin(c)

# # c = [sum(I) / (round(20*0.75, RoundDown)) for I in particles.index]


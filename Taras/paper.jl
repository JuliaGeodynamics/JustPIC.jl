using MAT
# using GLMakie

using JustPIC
using JustPIC._2D
const backend = JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

using JLD2, GLMakie

# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA"),
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU")
const backend = JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend


function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = x1 - dx
    xF = x2 + dx
    return range(xI, xF, length = n + 2)
end

A = matread("Taras/CornerFlow2D.mat")
const Vx_MAT = A["Vx"]
const Vy_MAT = A["Vy"]

function main(np, integrator)

    V = Vx_MAT, Vy_MAT

    lx = 100.0e3 # Horizontal model size, m
    ly = 100.0e3 # Vertical model size, m
    nx = 40    # Horizontal grid resolution
    ny = 40    # Vertical grid resolution
    nx_v = nx + 1
    ny_v = ny + 1
    dx = lx / (nx) # Horizontal grid step, m
    dy = ly / (ny) # Vertical grid step, m
    x = range(0, lx, length = nx_v)  # Horizontal coordinates of basic grid points, m
    y = range(0, ly, length = ny_v)  # Vertical coordinates of basic grid points, m

    # nodal centers
    xc, yc = range(0 + dx / 2, lx - dx / 2, length = nx), range(0 + dy / 2, ly - dy / 2, length = ny)
    # staggered grid velocity nodal locations
    grid_vx = x, expand_range(yc)
    grid_vy = expand_range(xc), y

    xvi = x, y

    grid_vxi = (
        grid_vx,
        grid_vy,
    )

    nxcell, max_xcell, min_xcell = np, 50, 1
    # nodal vertices
    xvi = x, y

    # dt = 2.2477e7
    particles1 = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi...,
    )
    particles2 = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi...,
    )
    particles3 = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi...,
    )

    dt = min(dx / maximum(abs.(Array(Vx_MAT))), dy / maximum(abs.(Array(Vy_MAT))))
    dt *= 0.75

    # ntime = 1000000
    # ntime      = 1000#00
    ntime = 50000 #0
    ntime = 10000 #0
    np_Bi = zeros(Int64, ntime)
    np_MQS = zeros(Int64, ntime)
    np_LinP = zeros(Int64, ntime)
    empty_Bi = zeros(Int64, ntime)
    empty_MQS = zeros(Int64, ntime)
    empty_LinP = zeros(Int64, ntime)
    full_Bi = zeros(Int64, ntime)
    full_MQS = zeros(Int64, ntime)
    full_LinP = zeros(Int64, ntime)

    time_Bi = zeros(ntime)
    time_MQS = zeros(ntime)
    time_LinP = zeros(ntime)

    for it in 1:ntime
        time_Bi0 = @elapsed advection!(particles1, integrator, V, grid_vxi, dt)
        time_MQS0 = @elapsed advection_LinP!(particles2, integrator, V, grid_vxi, dt)
        time_LinP0 = @elapsed advection_MQS!(particles3, integrator, V, grid_vxi, dt)

        time_Bi0 += @elapsed move_particles!(particles1, xvi, ())
        time_MQS0 += @elapsed move_particles!(particles2, xvi, ())
        time_LinP0 += @elapsed move_particles!(particles3, xvi, ())
        time_Bi[it] = time_Bi0
        time_MQS[it] = time_MQS0
        time_LinP[it] = time_LinP0

        # for (p, timer0) in zip( (particles1,particles2,particles3), (time_Bi0, time_MQS0, time_LinP0))
        #     timer0 += @elapsed move_particles!(p, xvi, ())
        #     push!(timer, timer0)
        #     # inject_particles!(p, (), xvi)
        # end
        # # inject && inject_particles!(particles, (), xvi)

        # total particles
        np_Bi[it] = sum(particles1.index.data)
        np_LinP[it] = sum(particles2.index.data)
        np_MQS[it] = sum(particles3.index.data)

        # empty
        empty_Bi[it] = sum([all(iszero, p) for p in particles1.index])
        empty_LinP[it] = sum([all(iszero, p) for p in particles2.index])
        empty_MQS[it] = sum([all(iszero, p) for p in particles3.index])

        # full
        full_Bi[it] = sum([sum(isone, p) > np for p in particles1.index])
        full_LinP[it] = sum([sum(isone, p) > np for p in particles2.index])
        full_MQS[it] = sum([sum(isone, p) > np for p in particles3.index])
    end
    stats_Bi = (; np = np_Bi, empty = empty_Bi, time = time_Bi, full = full_Bi)
    stats_LinP = (; np = np_LinP, empty = empty_LinP, time = time_MQS, full = full_LinP)
    stats_MQS = (; np = np_MQS, empty = empty_MQS, time = time_LinP, full = full_MQS)

    return particles1, particles2, particles3, stats_Bi, stats_LinP, stats_MQS
end

function runner()
    for integrator in (RungeKutta2(), RungeKutta4()), np in (4,)
        # for integrator in (RungeKutta2(), RungeKutta4()), np in (4,8,12,16,20,24)

        println("Sarting with np = $np...")
        particles1, particles2, particles3, stats_Lin, stats_LinP, stats_MQS = main(np, integrator)

        name = integrator == RungeKutta2() ? "RK2" : "RK4"

        jldsave(
            "Taras/data/CornerFlow2D_$(np)particles_NO_injection_$(name).jld2",
            particles1 = particles1,
            particles2 = particles2,
            particles3 = particles3,
            stats_Lin = stats_Lin,
            stats_LinP = stats_LinP,
            stats_MQS = stats_MQS
        )
        println("...done with np = $np")

    end
    return
end

runner()

let
    lx = 100.0e3 # Horizontal model size, m
    ly = 100.0e3 # Vertical model size, m
    nx = 40    # Horizontal grid resolution
    ny = 40    # Vertical grid resolution
    nx_v = nx + 1
    ny_v = ny + 1
    dx = lx / (nx) # Horizontal grid step, m
    dy = ly / (ny) # Vertical grid step, m
    x = range(0, lx, length = nx_v)  # Horizontal coordinates of basic grid points, m
    y = range(0, ly, length = ny_v)  # Vertical coordinates of basic grid points, m

    # nodal centers
    xc, yc = range(0 + dx / 2, lx - dx / 2, length = nx), range(0 + dy / 2, ly - dy / 2, length = ny)
  
    drk2 = jldopen("Taras/data/CornerFlow2D_4particles_NO_injection_RK2.jld2")
    drk4 = jldopen("Taras/data/CornerFlow2D_4particles_NO_injection_RK4.jld2")

    # particles1, particles2, particles3, stats_Lin, stats_LinP, stats_MQS = main(4, RungeKutta4())
    # particles1_rk2, particles2_rk2, particles3_rk2, stats_Lin, stats_LinP, stats_MQS = main(4, RungeKutta2())


    particles1_rk2, particles2_rk2, particles3_rk2 = jldopen("Taras/data/CornerFlow2D_4particles_NO_injection_RK2.jld2", "r") do file
        particles1_rk2 = file["particles1"]
        particles2_rk2 = file["particles2"]
        particles3_rk2 = file["particles3"]

        particles1_rk2, particles2_rk2, particles3_rk2
    end

    particles1_rk4, particles2_rk4, particles3_rk4 = jldopen("Taras/data/CornerFlow2D_4particles_NO_injection_RK4.jld2", "r") do file
        particles1_rk4 = file["particles1"]
        particles2_rk4 = file["particles2"]
        particles3_rk4 = file["particles3"]

        particles1_rk4, particles2_rk4, particles3_rk4
    end

    f = Figure(size = (1100, 1600))

    ax1 = Axis(f[1, 1], aspect = 1, subtitle = "Linear", subtitlesize = 24, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "x [km]", ylabelsize = 24, ylabel = "y [km]", title = "Runge-Kutta 2", titlesize = 30)
    ax2 = Axis(f[2, 1], aspect = 1, subtitle = "LinP", subtitlesize = 24, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "x [km]", ylabelsize = 24, ylabel = "y [km]")
    ax3 = Axis(f[3, 1], aspect = 1, subtitle = "MQS", subtitlesize = 24, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "x [km]", ylabelsize = 24, ylabel = "y [km]")
    ax4 = Axis(f[1, 2], aspect = 1, subtitle = "Linear", subtitlesize = 24, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "x [km]", ylabelsize = 24, ylabel = "y [km]", title = "Runge-Kutta 4", titlesize = 30)
    ax5 = Axis(f[2, 2], aspect = 1, subtitle = "LinP", subtitlesize = 24, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "x [km]", ylabelsize = 24, ylabel = "y [km]")
    ax6 = Axis(f[3, 2], aspect = 1, subtitle = "MQS", subtitlesize = 24, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "x [km]", ylabelsize = 24, ylabel = "y [km]")

    for (d, a) in zip((particles1_rk2, particles2_rk2, particles3_rk2), (ax1, ax2, ax3))
        scatter!(
            a,
            d.coords[1].data[:] ./ 1.0e3,
            d.coords[2].data[:] ./ 1.0e3,
            color = :black,
            markersize = 3,
        )
    end
    for (d, a) in zip((particles1_rk4, particles2_rk4, particles3_rk4), (ax4, ax5, ax6))
        scatter!(
            a,
            d.coords[1].data[:] ./ 1.0e3,
            d.coords[2].data[:] ./ 1.0e3,
            color = :black,
            markersize = 3,
        )
    end

    for a in (ax4, ax5, ax6)
        hideydecorations!(a; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
    end

    for a in (ax1, ax2, ax4, ax5)
        hidexdecorations!(a; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
    end

    f

    save("Taras/CornerFlow2D_particles_NO_injection.png", f)


    f = Figure(size = (1100, 1600))

    ax1 = Axis(f[1, 1], limits = (0, 30, 0, 30), aspect = 1, subtitle = "Linear", subtitlesize = 24, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "x [km]", ylabelsize = 24, ylabel = "y [km]", title = "Runge-Kutta 2", titlesize = 30)
    ax2 = Axis(f[2, 1], limits = (0, 30, 0, 30), aspect = 1, subtitle = "LinP", subtitlesize = 24, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "x [km]", ylabelsize = 24, ylabel = "y [km]")
    ax3 = Axis(f[3, 1], limits = (0, 30, 0, 30), aspect = 1, subtitle = "MQS", subtitlesize = 24, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "x [km]", ylabelsize = 24, ylabel = "y [km]")
    ax4 = Axis(f[1, 2], limits = (0, 30, 0, 30), aspect = 1, subtitle = "Linear", subtitlesize = 24, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "x [km]", ylabelsize = 24, ylabel = "y [km]", title = "Runge-Kutta 4", titlesize = 30)
    ax5 = Axis(f[2, 2], limits = (0, 30, 0, 30), aspect = 1, subtitle = "LinP", subtitlesize = 24, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "x [km]", ylabelsize = 24, ylabel = "y [km]")
    ax6 = Axis(f[3, 2], limits = (0, 30, 0, 30), aspect = 1, subtitle = "MQS", subtitlesize = 24, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "x [km]", ylabelsize = 24, ylabel = "y [km]")

    for (d, a) in zip((particles1_rk2, particles2_rk2, particles3_rk2), (ax1, ax2, ax3))
        scatter!(
            a,
            d.coords[1].data[:] ./ 1.0e3,
            d.coords[2].data[:] ./ 1.0e3,
            color = :black,
            markersize = 3,
        )
    end
    for (d, a) in zip((particles1_rk4, particles2_rk4, particles3_rk4), (ax4, ax5, ax6))
        scatter!(
            a,
            d.coords[1].data[:] ./ 1.0e3,
            d.coords[2].data[:] ./ 1.0e3,
            color = :black,
            markersize = 3,
        )
    end

    for a in (ax4, ax5, ax6)
        hideydecorations!(a; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
    end

    for a in (ax1, ax2, ax4, ax5)
        hidexdecorations!(a; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
    end
    f
    save("Taras/CornerFlow2D_particles_zoomed_NO_injection.png", f)

    f = Figure(size = (1100, 1600))

    ax1 = Axis(f[1, 1], aspect = 1, subtitle = "Linear", subtitlesize = 24, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "x [km]", ylabelsize = 24, ylabel = "y [km]", title = "Runge-Kutta 2", titlesize = 30)
    ax2 = Axis(f[2, 1], aspect = 1, subtitle = "LinP", subtitlesize = 24, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "x [km]", ylabelsize = 24, ylabel = "y [km]")
    ax3 = Axis(f[3, 1], aspect = 1, subtitle = "MQS", subtitlesize = 24, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "x [km]", ylabelsize = 24, ylabel = "y [km]")
    ax4 = Axis(f[1, 2], aspect = 1, subtitle = "Linear", subtitlesize = 24, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "x [km]", ylabelsize = 24, ylabel = "y [km]", title = "Runge-Kutta 4", titlesize = 30)
    ax5 = Axis(f[2, 2], aspect = 1, subtitle = "LinP", subtitlesize = 24, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "x [km]", ylabelsize = 24, ylabel = "y [km]")
    ax6 = Axis(f[3, 2], aspect = 1, subtitle = "MQS", subtitlesize = 24, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "x [km]", ylabelsize = 24, ylabel = "y [km]")

    cm = :vik
    for (d, a) in zip((particles1_rk2, particles2_rk2, particles3_rk2), (ax1, ax2, ax3))
        c = [count(i) for i in d.index]
        heatmap!(
            a,
            (xc, yc) ./ 1.0e3...,
            c,
            colormap = cm,
            markersize = 3,
        )
    end
    cl = 0
    for (d, a) in zip((particles1_rk4, particles2_rk4, particles3_rk4), (ax4, ax5, ax6))
        cl += 1
        c = [count(i) for i in d.index]
        h = heatmap!(
            a,
            (xc, yc) ./ 1.0e3...,
            c,
            colormap = cm,
            markersize = 3,
        )
        Colorbar(f[cl, 3], h, label = "particles per cell", labelsize = 24, ticklabelsize = 24)
    end

    for a in (ax4, ax5, ax6)
        hideydecorations!(a; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
    end

    for a in (ax1, ax2, ax4, ax5)
        hidexdecorations!(a; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
    end
    f
    save("Taras/CornerFlow2D_particles_heatmaps_injection.png", f)

    # stats_Bi   = (; np = np_Bi  , empty = empty_Bi  , time = time_Bi,   full = full_Bi)
    # stats_LinP = (; np = np_LinP, empty = empty_LinP, time = time_MQS,  full = full_LinP)
    # stats_MQS  = (; np = np_MQS , empty = empty_MQS , time = time_LinP, full = full_MQS)


    stats_Bi_rk2 = drk2["stats_Lin"]
    stats_LinP_rk2 = drk2["stats_LinP"]
    stats_MQS_rk2 = drk2["stats_MQS"]

    stats_Bi_rk4 = drk4["stats_Lin"]
    stats_LinP_rk4 = drk4["stats_LinP"]
    stats_MQS_rk4 = drk4["stats_MQS"]


    f = Figure(size = (1400, 1100))
    ax1 = Axis(f[1, 1], aspect = 1, subtitlesize = 24, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "timestep", ylabelsize = 24, ylabel = "full cells", title = "Runge-Kutta 2", titlesize = 30)
    ax2 = Axis(f[1, 2], aspect = 1, subtitlesize = 24, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "timestep", ylabelsize = 24, ylabel = "full cells", title = "Runge-Kutta 4", titlesize = 30)

    lw = 3
    lines!(ax1, stats_Bi_rk2.full, label = "Linear", linewidth = lw)
    lines!(ax1, stats_LinP_rk2.full, label = "LinP", linewidth = lw)
    lines!(ax1, stats_MQS_rk2.full, label = "MQS", linewidth = lw)
    lines!(ax2, stats_Bi_rk4.full, label = "Linear", linewidth = lw)
    lines!(ax2, stats_LinP_rk4.full, label = "LinP", linewidth = lw)
    lines!(ax2, stats_MQS_rk4.full, label = "MQS", linewidth = lw)

    linkaxes!(ax1, ax2)
    axislegend(ax1, position = :lb, labelsize = 24)
    hideydecorations!(ax2; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
    f
    save("Taras/CornerFlow2D_particles_NO_injection_full_cells.png", f)


    f = Figure(size = (1400, 1100))
    ax1 = Axis(f[1, 1], aspect = 1, subtitlesize = 24, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "timestep", ylabelsize = 24, ylabel = "empty cells", title = "Runge-Kutta 2", titlesize = 30)
    ax2 = Axis(f[1, 2], aspect = 1, subtitlesize = 24, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "timestep", ylabelsize = 24, ylabel = "empty cells", title = "Runge-Kutta 4", titlesize = 30)

    lw = 3
    lines!(ax1, stats_Bi_rk2.empty, label = "Linear", linewidth = lw)
    lines!(ax1, stats_LinP_rk2.empty, label = "LinP", linewidth = lw)
    lines!(ax1, stats_MQS_rk2.empty, label = "MQS", linewidth = lw)
    lines!(ax2, stats_Bi_rk4.empty, label = "Linear", linewidth = lw)
    lines!(ax2, stats_LinP_rk4.empty, label = "LinP", linewidth = lw)
    lines!(ax2, stats_MQS_rk4.empty, label = "MQS", linewidth = lw)

    linkaxes!(ax1, ax2)
    axislegend(ax1, position = :lt, labelsize = 24)
    hideydecorations!(ax2; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)

    f
    save("Taras/CornerFlow2D_particles_NO_injection_empty_cells.png", f)

    f = Figure(size = (1400, 1100))
    ax1 = Axis(f[1, 1], aspect = 1, subtitlesize = 24, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "timestep", ylabelsize = 24, ylabel = "accumulated wall time [s]", title = "Runge-Kutta 2", titlesize = 30)
    ax2 = Axis(f[1, 2], aspect = 1, subtitlesize = 24, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "timestep", ylabelsize = 24, ylabel = "accumulated wall time [s]", title = "Runge-Kutta 4", titlesize = 30)

    lw = 4
    lines!(ax1, cumsum(stats_Bi_rk2.time), label = "Linear", linewidth = lw)
    lines!(ax1, cumsum(stats_LinP_rk2.time), label = "LinP", linewidth = lw)
    lines!(ax1, cumsum(stats_MQS_rk2.time), label = "MQS", linewidth = lw)
    lines!(ax2, cumsum(stats_Bi_rk4.time), label = "Linear", linewidth = lw)
    lines!(ax2, cumsum(stats_LinP_rk4.time), label = "LinP", linewidth = lw)
    lines!(ax2, cumsum(stats_MQS_rk4.time), label = "MQS", linewidth = lw)

    linkaxes!(ax1, ax2)
    axislegend(ax1, position = :lt, labelsize = 24)
    hideydecorations!(ax2; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)

    f

    save("Taras/CornerFlow2D_particles_NO_injection_cumsum_time.png", f)
end

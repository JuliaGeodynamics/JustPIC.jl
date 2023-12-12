using CSV, DataFrames, LinearRegression
using GLMakie
# using CairoMakie

getname(d, n, np; pth="timings") = joinpath(pth, (d[n])[np])

nt = 6
fldr = "timings/nt$(nt)/"

n = 32, 64, 128#, 256
np = 6,12,18,24
# organise files
d = Dict{Int, Dict{Int, String}}(
    n => Dict{Int, String}(
        np => "timings_n$(n)_np$(np)" for np in np
    ) 
    for n in n
)

c = [:orangered, :goldenrod2, :chartreuse4, :deeppink4]

fig = Figure(size = (1500, 800))
ax = [Axis(fig[1, i]) for i in eachindex(n)]

for (j,res) in enumerate(n)
    fname6  = getname(d, res, 6 ; pth = fldr)
    fname12 = getname(d, res, 12; pth = fldr)
    fname18 = getname(d, res, 18; pth = fldr)
    fname24 = getname(d, res, 24; pth = fldr)
    dt6  = CSV.read(fname6 * ".csv", DataFrame)
    dt12 = CSV.read(fname12 * ".csv", DataFrame)
    dt18 = CSV.read(fname18 * ".csv", DataFrame)
    dt24 = CSV.read(fname24 * ".csv", DataFrame)

    t_opt = dt6.opt, dt12.opt, dt18.opt, dt24.opt 
    t_classic = dt6.classic, dt12.classic, dt18.classic, dt24.classic
    nt = length(t_opt[1])
    x = LinRange(0, 0.01, nt-1)

    for i in 1:4
        scatter!(ax[j], x, t_opt[i][2:end], color=c[i], label = "$(np[i])")
        scatter!(ax[j], x, t_classic[i][2:end], color=c[i], marker=:diamond)
    end
    if j == 1
        ax[j].ylabel = "wall-time [s]"
        # axislegend("particles per cell")
    end

    ax2 = Axis(fig[2, j])
    for i in 1:4
        scatter!(ax2, x, t_opt[i][2:end] ./t_classic[i][2:end] , color=c[i])
    end
    ax2.xlabel = "iteration"
    ax2.ylabel = "wall-time opt / classic "
    ylims!(ax2, 0, 1.5)

end
fig

############

fig = Figure(size = (1500, 450))
ax = [Axis(fig[1, i]) for i in eachindex(n)]

for (j,res) in enumerate(n)
    fname6  = getname(d, res, 6 ; pth = fldr)
    fname12 = getname(d, res, 12; pth = fldr)
    fname18 = getname(d, res, 18; pth = fldr)
    fname24 = getname(d, res, 24; pth = fldr)
    dt6  = CSV.read(fname6 * ".csv", DataFrame)
    dt12 = CSV.read(fname12 * ".csv", DataFrame)
    dt18 = CSV.read(fname18 * ".csv", DataFrame)
    dt24 = CSV.read(fname24 * ".csv", DataFrame)

    t_opt = dt6.adv_opt, dt12.adv_opt, dt18.adv_opt, dt24.adv_opt 
    t_classic = dt6.adv_classic, dt12.adv_classic, dt18.adv_classic, dt24.adv_classic
    nt = length(t_opt[1])
    x = LinRange(0, 0.04, nt-1)

    for i in 1:4
        scatter!(ax[j], x, t_opt[i][2:end]./t_classic[i][2:end], color=c[i], label = "$(np[i])")
        # scatter!(ax[j], x, t_classic[i][2:end], color=c[i], marker=:diamond)
    end
    if j == 1
        # ax[j].ylabel = "wall-time [s]"
        ax[j].ylabel = "wall-time opt / classic "
        # axislegend("particles per cell")
    end
    ylims!(ax[j], 0, 1)
    ax[j].xlabel = "iteration"
end
fig


############

fig = Figure(size = (1500, 450))
ax = [Axis(fig[1, i]) for i in eachindex(n)]

for (j,res) in enumerate(n)
    fname6  = getname(d, res, 6 ; pth = fldr)
    fname12 = getname(d, res, 12; pth = fldr)
    fname18 = getname(d, res, 18; pth = fldr)
    fname24 = getname(d, res, 24; pth = fldr)
    dt6  = CSV.read(fname6 * ".csv", DataFrame)
    dt12 = CSV.read(fname12 * ".csv", DataFrame)
    dt18 = CSV.read(fname18 * ".csv", DataFrame)
    dt24 = CSV.read(fname24 * ".csv", DataFrame)

    t_opt = dt6.g2p_opt, dt12.g2p_opt, dt18.g2p_opt, dt24.g2p_opt 
    t_classic = dt6.g2p_classic, dt12.g2p_classic, dt18.g2p_classic, dt24.g2p_classic
    nt = length(t_opt[1])
    x = LinRange(0, 0.04, nt-1)

    for i in 1:4
        scatter!(ax[j], x, t_opt[i][2:end]./t_classic[i][2:end], color=c[i], label = "$(np[i])")
        # scatter!(ax[j], x, t_classic[i][2:end], color=c[i], marker=:diamond)
    end
    if j == 1
        # ax[j].ylabel = "wall-time [s]"
        ax[j].ylabel = "wall-time opt / classic "
        # axislegend("particles per cell")
    end
    ylims!(ax[j], 0, 3)
    ax[j].xlabel = "iteration"
end
fig


############

fig = Figure(size = (1500, 450))
ax = [Axis(fig[1, i]) for i in eachindex(n)]

for (j,res) in enumerate(n)
    fname6  = getname(d, res, 6 ; pth = fldr)
    fname12 = getname(d, res, 12; pth = fldr)
    fname18 = getname(d, res, 18; pth = fldr)
    fname24 = getname(d, res, 24; pth = fldr)
    dt6  = CSV.read(fname6 * ".csv", DataFrame)
    dt12 = CSV.read(fname12 * ".csv", DataFrame)
    dt18 = CSV.read(fname18 * ".csv", DataFrame)
    dt24 = CSV.read(fname24 * ".csv", DataFrame)

    t_opt = dt6.p2g_opt, dt12.p2g_opt, dt18.p2g_opt, dt24.p2g_opt 
    t_classic = dt6.p2g_classic, dt12.p2g_classic, dt18.p2g_classic, dt24.p2g_classic
    nt = length(t_opt[1])
    x = LinRange(0, 0.04, nt-1)

    for i in 1:4
        scatter!(ax[j], x, t_opt[i][2:end]./t_classic[i][2:end], color=c[i], label = "$(np[i])")
        # scatter!(ax[j], x, t_classic[i][2:end], color=c[i], marker=:diamond)
    end
    if j == 1
        # ax[j].ylabel = "wall-time [s]"
        ax[j].ylabel = "wall-time opt / classic "
        # axislegend("particles per cell")
    end
    ylims!(ax[j], 0, 1)
    ax[j].xlabel = "iteration"
end
fig

# save("timings2.pdf", fig)

# l = ["physical advection", "grid → p", "particle → grid", "memory advection"]
# data_opt = [
#     mean(dt6.adv_opt),
#     mean(dt6.g2p_opt),
#     mean(dt6.p2g_opt),
#     mean(dt6.shuffle),
# ]
# data_classic = [
#     mean(dt6.adv_classic),
#     mean(dt6.g2p_classic),
#     mean(dt6.p2g_classic),
# ]

# fpie = Figure(size = (900, 900))
# ax = Axis(fpie[1, 1], autolimitaspect = 1)
# pie!(
#     ax,
#     data_opt,
#     # normalize = false, 
#     color = c,
#     radius = 4,
#     inner_radius = 2,
#     strokecolor = :white,
#     strokewidth = 5,
#     # axis = (autolimitaspect = 1, ),
#     label = l
# )
# hidedecorations!(ax)
# hidespines!(ax)
# ax = Axis(fpie[2, 1], autolimitaspect = 1)
# pie!(
#     ax,
#     data_classic ./ mean(dt6.opt), 
#     # normalize = false,
#     color = c[1:end-1],
#     radius = 4,
#     inner_radius = 2,
#     strokecolor = :white,
#     strokewidth = 5,
#     # axis = (autolimitaspect = 1, ),
#     label = l
# )
# hidedecorations!(ax)
# hidespines!(ax)
# Legend(
#     fpie[:,2], 
#     [PolyElement(color=c) for c in c], 
#     l, 
#     framevisible=false)
# fpie
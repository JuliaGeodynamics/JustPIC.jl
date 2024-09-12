using CSV, DataFrames
using GLMakie
# using CairoMakie

getname(d, n, np; pth="timings_Daint") = joinpath(pth, (d[n])[np])
# save_path = "C:\\Users\\albert\\Desktop\\Manuscripts\\ParticlesGMD\\figs\\"

let 
    for nt in  (6,)

        save_path = (@__DIR__)

        fldr = joinpath(@__DIR__, "timings_gpu")
        # fldr = joinpath(@__DIR__, "timings_notrand_cpu/nt$(nt)")
        n = [
            256 1024
            512 2048
        ]
        # n = [
        #     16  64
        #     32 128 
        # ]
        np = (6, 12, 18, 24)
        # organise files
        d = ["timings_gpu/cutimers_$(n)x$(n)_nt_6" for n in n]

        c = [:orangered, :goldenrod2, :chartreuse4, :deepskyblue4]
        l = [string(n) for n in np]

        fig = Figure(size = (1500, 800))
        ax = [Axis(fig[i, j], title="$(n[i,j]) x $(n[i,j])") for i in 1:2, j in 1:2]

        cols = 3, 4, 5, 6
        for (j, res) in enumerate(n)
            fname24 = d[j]
            dt24    = CSV.read(fname24 * ".csv", DataFrame) 
            
            t_opt = dt24.opt 
            # t_opt = 
            #     sum(dt6[!,i]  for i in cols),
            #     sum(dt12[!,i] for i in cols),
            #     sum(dt18[!,i] for i in cols),
            #     sum(dt24[!,i] for i in cols)
             
            t_classic = dt6.classic, dt12.classic, dt18.classic, dt24.classic 
            nt = length(t_opt[1])
            x = LinRange(0, 0.04, nt-1)

            for i in 1:4
                # scatter!(ax[j], x, t_opt[i][2:end], color=c[i], label = "$(np[i])")
                # scatter!(ax[j], x, t_classic[i][2:end], color=c[i], marker=:diamond)
                scatter!(ax[j], x, t_opt[i][2:end] ./t_classic[i][2:end] , color=c[i], markersize= 4, label=l[i])
                hlines!(ax[j], 1.0, color=:black)
                ylims!(ax[j], 0, 1.5)
                xlims!(ax[j], 0, 0.04)
            end
        end

        fig[:, 3] = Legend(fig, ax[2, 2], "Particles per cell (t=0)", framevisible = true)

        ylims!(ax[2,1], 0, 1.5)
        ylims!(ax[2,2], 0, 1.5)
        ax[1, 1].ylabel = "wall-time opt / classic"
        ax[2, 1].ylabel = "wall-time opt / classic"
        hidexdecorations!(ax[1, 1]; 
            label = true, ticklabels = true, ticks = true,
            grid = false, minorgrid = false, minorticks = false
        )
        hidexdecorations!(ax[1, 2]; 
            label = true, ticklabels = true, ticks = true,
            grid = false, minorgrid = false, minorticks = false
        )
        hideydecorations!(ax[1, 2]; 
            label = true, ticklabels = true, ticks = true,
            grid = false, minorgrid = false, minorticks = false
        )
        hideydecorations!(ax[2, 2]; 
            label = true, ticklabels = true, ticks = true,
            grid = false, minorgrid = false, minorticks = false
        )
        fig
        save(save_path*"timings_cpu_new.png", fig)

        ############

        fig = Figure(size = (1500, 800))
        ax = [Axis(fig[i, j], title="$(n[i,j]) x $(n[i,j])") for i in 1:2, j in 1:2]

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
                scatter!(ax[j], x, t_opt[i][2:end]./t_classic[i][2:end], color=c[i], markersize= 4, label = "$(np[i])")
                hlines!(ax[j], 1.0, color=:black)
            end
            xlims!(ax[j], 0, 0.04)
            ylims!(ax[j], 0, 1.5)
            ax[j].xlabel = "iteration"
        end

        fig[:, 3] = Legend(fig, ax[2, 2], "Particles per cell (t=0)", framevisible = true)
        ax[1, 1].ylabel = "wall-time opt / classic"
        ax[2, 1].ylabel = "wall-time opt / classic"
        hidexdecorations!(ax[1, 1]; 
            label = true, ticklabels = true, ticks = true,
            grid = false, minorgrid = false, minorticks = false
        )
        hidexdecorations!(ax[1, 2]; 
            label = true, ticklabels = true, ticks = true,
            grid = false, minorgrid = false, minorticks = false
        )
        hideydecorations!(ax[1, 2]; 
            label = true, ticklabels = true, ticks = true,
            grid = false, minorgrid = false, minorticks = false
        )
        hideydecorations!(ax[2, 2]; 
            label = true, ticklabels = true, ticks = true,
            grid = false, minorgrid = false, minorticks = false
        )
        fig
        save(save_path*"timings_cpu_advection_new.png", fig)

        ############

        fig = Figure(size = (1500, 800))
        ax = [Axis(fig[i, j], title="$(n[i,j]) x $(n[i,j])") for i in 1:2, j in 1:2]

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
                scatter!(ax[j], x, t_opt[i][2:end]./t_classic[i][2:end], color=c[i], markersize= 4, label = "$(np[i])")
                hlines!(ax[j], 1.0, color=:black)
            end
            ylims!(ax[j], 0, 1.5)
            xlims!(ax[j], 0, 0.04)
            ax[j].xlabel = "iteration"
        end
        fig[:, 3] = Legend(fig, ax[2, 2], "Particles per cell (t=0)", framevisible = true)
        ax[1, 1].ylabel = "wall-time opt / classic"
        ax[2, 1].ylabel = "wall-time opt / classic"
        hidexdecorations!(ax[1, 1]; 
            label = true, ticklabels = true, ticks = true,
            grid = false, minorgrid = false, minorticks = false
        )
        hidexdecorations!(ax[1, 2]; 
            label = true, ticklabels = true, ticks = true,
            grid = false, minorgrid = false, minorticks = false
        )
        hideydecorations!(ax[1, 2]; 
            label = true, ticklabels = true, ticks = true,
            grid = false, minorgrid = false, minorticks = false
        )
        hideydecorations!(ax[2, 2]; 
            label = true, ticklabels = true, ticks = true,
            grid = false, minorgrid = false, minorticks = false
        )
        fig
        save(save_path*"timings_cpu_g2p_new.png", fig)

        ############

        fig = Figure(size = (1500, 800))
        ax = [Axis(fig[i, j], title="$(n[i,j]) x $(n[i,j])") for i in 1:2, j in 1:2]

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
                scatter!(ax[j], x, t_opt[i][2:end]./t_classic[i][2:end], color=c[i], markersize= 4, label = "$(np[i])")
                hlines!(ax[j], 1.0, color=:black)
            end
            ylims!(ax[j], 0, 1.5)
            xlims!(ax[j], 0, 0.04)
            ax[j].xlabel = "iteration"
        end
        fig[:, 3] = Legend(fig, ax[2, 2], "Particles per cell (t=0)", framevisible = true)
        ax[1, 1].ylabel = "wall-time opt / classic"
        ax[2, 1].ylabel = "wall-time opt / classic"
        hidexdecorations!(ax[1, 1]; 
            label = true, ticklabels = true, ticks = true,
            grid = false, minorgrid = false, minorticks = false
        )
        hidexdecorations!(ax[1, 2]; 
            label = true, ticklabels = true, ticks = true,
            grid = false, minorgrid = false, minorticks = false
        )
        hideydecorations!(ax[1, 2]; 
            label = true, ticklabels = true, ticks = true,
            grid = false, minorgrid = false, minorticks = false
        )
        hideydecorations!(ax[2, 2]; 
            label = true, ticklabels = true, ticks = true,
            grid = false, minorgrid = false, minorticks = false
        )
        fig
        save(save_path*"timings_cpu_p2g_new.png", fig)
    end
end

# cols = 3, 4, 5, 6
# # for (j, res) in enumerate(n)
#     fname6  = getname(d, res, 6 ; pth = fldr)
#     fname12 = getname(d, res, 12; pth = fldr)
#     fname18 = getname(d, res, 18; pth = fldr)
#     fname24 = getname(d, res, 24; pth = fldr)
#     dt      = (
#         CSV.read(fname6 * ".csv", DataFrame),
#         CSV.read(fname12 * ".csv", DataFrame),
#         CSV.read(fname18 * ".csv", DataFrame),
#         CSV.read(fname24 * ".csv", DataFrame),
#     )
# # end

# # classic
# adv_cl = ntuple(Val(4)) do i
#     dt[i].adv_classic |> mean
# end

# p2g_cl = ntuple(Val(4)) do i
#     dt[i].p2g_classic |> mean
# end

# g2p_cl = ntuple(Val(4)) do i
#     dt[i].g2p_classic |> mean 
# end

# ## opt
# adv_opt = ntuple(Val(4)) do i
#     dt[i].adv_opt |> mean 
# end

# p2g_opt = ntuple(Val(4)) do i
#     dt[i].p2g_opt |> mean 
# end

# g2p_opt = ntuple(Val(4)) do i
#     dt[i].g2p_opt |> mean 
# end

# move_opt = ntuple(Val(4)) do i
#     dt[i].shuffle |> mean 
# end


# fig = Figure()
# ax1 = Axis(fig[1,1], aspect=DataAspect())
# ax2 = Axis(fig[2,1], aspect=DataAspect())

# colors = Makie.wong_colors()[1:4];
# labels = ["advection", "p2g", "g2p", "mem shuffle"]
# pie!(
#     ax1,
#     [adv_opt[1], p2g_opt[1], g2p_opt[1], move_opt[1]],
#     color=colors,
#     strokecolor = :white,
#     strokewidth = 5,
# )
# Legend(fig[1,2], [PolyElement(color=c) for c in colors], labels, framevisible=false)

# colors = Makie.wong_colors()[1:3];
# labels = ["advection", "p2g", "g2p"]
# pie!(
#     ax2,
#     [adv_cl[1], p2g_cl[1], g2p_cl[1]],
#     color=colors,
#     strokecolor = :white,
#     strokewidth = 5,
# )

# Legend(fig[2,2], [PolyElement(color=c) for c in colors], labels, framevisible=false)
# fig
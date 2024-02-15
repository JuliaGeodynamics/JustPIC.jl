using CSV, DataFrames
using GLMakie
# using CairoMakie

getname(d, n, np; pth="timings_CUDA") = joinpath(pth, (d[n])[np])

let 

        save_path = (@__DIR__)
        fldr = joinpath(@__DIR__, "timings_CUDA")
        n = [
            64  256
            128 256
        ]
        np = 6, 12, 18, 24
        # organise files
        d = Dict{Int, Dict{Int, String}}(
            n => Dict{Int, String}(
                np => "timings_n$(n)_np$(np)" for np in np
            ) 
            for n in n
        )

        c = [:orangered, :goldenrod2, :chartreuse4, :deepskyblue4]
        l = [string(n) for n in np]

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

            t_opt = dt6.opt, dt12.opt, dt18.opt, dt24.opt 
            t_classic = dt6.classic, dt12.classic, dt18.classic, dt24.classic 
            nt = length(t_opt[1])
            x = LinRange(0, 0.04, nt-1)

            for i in 1:3
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
        save("timings_CUDA.png", fig)

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

            for i in 1:3
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
        save("timings_CUDA_advection.png", fig)

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

            for i in 1:3
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
        save("timings_CUDA_g2p.png", fig)

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

            for i in 1:3
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
        save("timings_CUDA_p2g.png", fig)
end
#### LOAD DATA

drk2     = jldopen("Taras/data/CornerFlow2D_6particles_NO_injection_RK2.jld2")
drk4     = jldopen("Taras/data/CornerFlow2D_6particles_NO_injection_RK4.jld2")
drk2_inj = jldopen("Taras/data/CornerFlow2D_6particles_injection_RK2.jld2")
drk4_inj = jldopen("Taras/data/CornerFlow2D_6particles_injection_RK4.jld2")

#### PLOT MARKERS

function plot_markers(drk2, drk4)
    f = Figure(size=(1100, 1600))

    ax1 = Axis(f[1, 1], aspect = 1, subtitle = "Bilinear", subtitlesize = 24, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "x [km]", ylabelsize = 24, ylabel = "y [km]", title = "Runge-Kutta 2" , titlesize = 30)
    ax2 = Axis(f[2, 1], aspect = 1, subtitle = "LinP"    , subtitlesize = 24, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "x [km]", ylabelsize = 24, ylabel = "y [km]")
    ax3 = Axis(f[3, 1], aspect = 1, subtitle = "MQS"     , subtitlesize = 24, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "x [km]", ylabelsize = 24, ylabel = "y [km]")
    ax4 = Axis(f[1, 2], aspect = 1, subtitle = "Bilinear", subtitlesize = 24, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "x [km]", ylabelsize = 24, ylabel = "y [km]", title = "Runge-Kutta 4" , titlesize = 30)
    ax5 = Axis(f[2, 2], aspect = 1, subtitle = "LinP"    , subtitlesize = 24, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "x [km]", ylabelsize = 24, ylabel = "y [km]")
    ax6 = Axis(f[3, 2], aspect = 1, subtitle = "MQS"     , subtitlesize = 24, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "x [km]", ylabelsize = 24, ylabel = "y [km]")


    p1 = drk2["particles1"];
    p2 = drk2["particles2"];
    p3 = drk2["particles3"];

    for (d, a) in zip( (p1, p2, p3), (ax1, ax2, ax3))
        scatter!(
            a,
            d.coords[1].data[:],
            d.coords[2].data[:],
            color = :black,
            markersize = 3,
        )
    end

    p1 = drk4["particles1"];
    p2 = drk4["particles2"];
    p3 = drk4["particles3"];

    for (d, a) in zip( (p1, p2, p3), (ax4, ax5, ax6))
        scatter!(
            a,
            d.coords[1].data[:],
            d.coords[2].data[:],
            color = :black,
            markersize = 3,
        )
    end

    for a in  (ax4, ax5, ax6)
        hideydecorations!(a; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
    end

    for a in  (ax1, ax2, ax4, ax5)
        hidexdecorations!(a; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
    end

    f
end

plot_markers(drk2, drk4)
# plot_markers(drk2_inj, drk4_inj)

# function plot_markers(drk2, drk4)
#     f = Figure(size=(1100, 1600))

#     ax1 = Axis(f[1, 1], aspect = 1, subtitle = "Bilinear", subtitlesize = 24, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "x [km]", ylabelsize = 24, ylabel = "y [km]", title = "Runge-Kutta 2" , titlesize = 30)
#     ax2 = Axis(f[2, 1], aspect = 1, subtitle = "LinP"    , subtitlesize = 24, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "x [km]", ylabelsize = 24, ylabel = "y [km]")
#     ax3 = Axis(f[3, 1], aspect = 1, subtitle = "MQS"     , subtitlesize = 24, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "x [km]", ylabelsize = 24, ylabel = "y [km]")
#     ax4 = Axis(f[1, 2], aspect = 1, subtitle = "Bilinear", subtitlesize = 24, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "x [km]", ylabelsize = 24, ylabel = "y [km]", title = "Runge-Kutta 4" , titlesize = 30)
#     ax5 = Axis(f[2, 2], aspect = 1, subtitle = "LinP"    , subtitlesize = 24, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "x [km]", ylabelsize = 24, ylabel = "y [km]")
#     ax6 = Axis(f[3, 2], aspect = 1, subtitle = "MQS"     , subtitlesize = 24, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "x [km]", ylabelsize = 24, ylabel = "y [km]")


#     p1 = drk2["particles1"];
#     p2 = drk2["particles2"];
#     p3 = drk2["particles3"];

#     for (d, a) in zip( (p1, p2, p3), (ax1, ax2, ax3))
#         heatmap!(
#             a,

#             d.coords[1].data[:],
#             d.coords[2].data[:],
#             color = :black,
#             markersize = 3,
#         )
#     end

#     p1 = drk4["particles1"];
#     p2 = drk4["particles2"];
#     p3 = drk4["particles3"];

#     for (d, a) in zip( (p1, p2, p3), (ax4, ax5, ax6))
#         scatter!(
#             a,
#             d.coords[1].data[:],
#             d.coords[2].data[:],
#             color = :black,
#             markersize = 3,
#         )
#     end

#     for a in  (ax4, ax5, ax6)
#         hideydecorations!(a; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
#     end

#     for a in  (ax1, ax2, ax4, ax5)
#         hidexdecorations!(a; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
#     end

#     f
# end

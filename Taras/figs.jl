using GLMakie, JLD2

function num_part(file)
    left  = findfirst("_", file)[1]
    right = findfirst("p", file)[1]
    np = parse(Int64, file[left+1:right-1])
end

pth = @__DIR__

files = filter(x->contains(x, "inj.jld2"), readdir(pth, join=true))
nps = num_part.(files)

isort = sortperm(nps)
files = files[isort]
nps = nps[isort]
data = jldopen.(files);

# d = data[1];
# particles1, particles2, particles3, stats_Lin, stats_LinP, stats_MQS = d["particles1"], d["particles2"], d["particles3"], d["stats_Lin"], d["stats_LinP"], d["stats_MQS"];

# d1 = [count(p) for p in particles1.index];
# d2 = [count(p) for p in particles2.index];
# d3 = [count(p) for p in particles3.index];

# heatmap(d1 ./ 8, colorrange = (0, 2), colormap=:lipari)
# heatmap(d2 ./ 8, colorrange = (0, 2), colormap=:lipari)
# heatmap(d3 ./ 8, colorrange = (0, 2), colormap=:lipari)

# f, ax, s = scatterlines(stats_Lin.np , markersize = 4, label = "bilinear")
# scatterlines!(ax, stats_LinP.np, markersize = 4, label = "LinP")
# scatterlines!(ax, stats_MQS.np , markersize = 4, label = "MQS")
# axislegend(ax)
# f

###### PLOT EMPTY CELLS
f = Figure(size=(1600, 1600))
ax1 = Axis(f[1, 1], title = "4 particles per cell" , titlesize = 30, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "time step", ylabelsize = 24, ylabel = "Percentage of empty cells (%)")
ax2 = Axis(f[2, 1], title = "12 particles per cell", titlesize = 30, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "time step", ylabelsize = 24, ylabel = "Percentage of empty cells (%)")
ax3 = Axis(f[1, 2], title = "16 particles per cell", titlesize = 30, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "time step", ylabelsize = 24, ylabel = "Percentage of empty cells (%)")
ax4 = Axis(f[2, 2], title = "20 particles per cell", titlesize = 30, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "time step", ylabelsize = 24, ylabel = "Percentage of empty cells (%)")

ax = ax1, ax2, ax3, ax4
for (i, ax) in enumerate(ax)
    d = data[i]
    particles1, particles2, particles3, stats_Lin, stats_LinP, stats_MQS = d["particles1"], d["particles2"], d["particles3"], d["stats_Lin"], d["stats_LinP"], d["stats_MQS"];

    scatter!(ax, 100 * stats_Lin.empty  ./ 40^2, label = "Bilinear")
    scatter!(ax, 100 * stats_LinP.empty ./ 40^2, label = "LinP")
    scatter!(ax, 100 * stats_MQS.empty  ./ 40^2, label = "MQS")
end
linkaxes!(ax...)
hidexdecorations!(ax3; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
hidexdecorations!(ax1; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
hideydecorations!(ax3; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
hideydecorations!(ax4; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
# axislegend(ax)
f[:, 3] = Legend(f, ax1, title = "Advection method", framevisible = false, labelsize=30, titlesize=36)
f


###### PLOT FULL CELLS
f = Figure(size=(1600, 1600))
ax1 = Axis(f[1, 1], title = "4 particles per cell" , titlesize = 30, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "time step", ylabelsize = 24, ylabel = "Percentage of full cells (%)")
ax2 = Axis(f[2, 1], title = "12 particles per cell", titlesize = 30, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "time step", ylabelsize = 24, ylabel = "Percentage of full cells (%)")
ax3 = Axis(f[1, 2], title = "16 particles per cell", titlesize = 30, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "time step", ylabelsize = 24, ylabel = "Percentage of full cells (%)")
ax4 = Axis(f[2, 2], title = "20 particles per cell", titlesize = 30, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "time step", ylabelsize = 24, ylabel = "Percentage of full cells (%)")

ax = ax1, ax2, ax3, ax4
for (i, ax) in enumerate(ax)
    d = data[i]
    particles1, particles2, particles3, stats_Lin, stats_LinP, stats_MQS = d["particles1"], d["particles2"], d["particles3"], d["stats_Lin"], d["stats_LinP"], d["stats_MQS"];

    scatter!(ax, 100 * stats_Lin.full  ./ 40^2, label = "Bilinear")
    scatter!(ax, 100 * stats_LinP.full ./ 40^2, label = "LinP")
    scatter!(ax, 100 * stats_MQS.full  ./ 40^2, label = "MQS")
end
linkaxes!(ax...)
hidexdecorations!(ax3; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
hidexdecorations!(ax1; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
hideydecorations!(ax3; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
hideydecorations!(ax4; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
# axislegend(ax)
f[:, 3] = Legend(f, ax1, title = "Advection method", framevisible = false, labelsize=30, titlesize=36)
f


###### PLOT TIMINGS
f = Figure(size=(1600, 1600))
ax1 = Axis(f[1, 1], title = "4 particles per cell" , titlesize = 30, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "time step", ylabelsize = 24, ylabel = "accumulated run time")
ax2 = Axis(f[2, 1], title = "12 particles per cell", titlesize = 30, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "time step", ylabelsize = 24, ylabel = "accumulated run time")
ax3 = Axis(f[1, 2], title = "16 particles per cell", titlesize = 30, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "time step", ylabelsize = 24, ylabel = "accumulated run time")
ax4 = Axis(f[2, 2], title = "20 particles per cell", titlesize = 30, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "time step", ylabelsize = 24, ylabel = "accumulated run time")

ax = ax1, ax2, ax3, ax4
for (i, ax) in enumerate(ax)
    d = data[i]
    particles1, particles2, particles3, stats_Lin, stats_LinP, stats_MQS = d["particles1"], d["particles2"], d["particles3"], d["stats_Lin"], d["stats_LinP"], d["stats_MQS"];

    scatter!(ax, stats_Lin.time  |> cumsum , label = "Bilinear")
    scatter!(ax, stats_LinP.time |> cumsum , label = "LinP")
    scatter!(ax, stats_MQS.time  |> cumsum , label = "MQS")
end
linkaxes!(ax...)
hidexdecorations!(ax3; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
hidexdecorations!(ax1; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
hideydecorations!(ax3; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
hideydecorations!(ax4; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
# axislegend(ax)
f[:, 3] = Legend(f, ax1, title = "Advection method", framevisible = false, labelsize=30, titlesize=36)
f

###### PLOT TIMINGS
f = Figure(size=(1600, 1600))
ax1 = Axis(f[1, 1], title = "4 particles per cell" , titlesize = 30, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "time step", ylabelsize = 24, ylabel = "accumulated run time")
ax2 = Axis(f[2, 1], title = "12 particles per cell", titlesize = 30, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "time step", ylabelsize = 24, ylabel = "accumulated run time")
ax3 = Axis(f[1, 2], title = "16 particles per cell", titlesize = 30, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "time step", ylabelsize = 24, ylabel = "accumulated run time")
ax4 = Axis(f[2, 2], title = "20 particles per cell", titlesize = 30, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, xlabel = "time step", ylabelsize = 24, ylabel = "accumulated run time")

ax = ax1, ax2, ax3, ax4
for (i, ax) in enumerate(ax)
    d = data[i]
    particles1, particles2, particles3, stats_Lin, stats_LinP, stats_MQS = d["particles1"], d["particles2"], d["particles3"], d["stats_Lin"], d["stats_LinP"], d["stats_MQS"];

    # scatter!(ax, stats_Lin.time  |> cumsum , label = "Bilinear")
    lines!(ax, cumsum(stats_LinP.time )./ cumsum(stats_Lin.time) , color = :black, linewidth = 3, linestyle = :solid, label = "LinP")
    lines!(ax, cumsum(stats_MQS.time  )./ cumsum(stats_Lin.time) , color = :black, linewidth = 3, linestyle = :dash, label = "MQS")
    ylims!(ax, 0.8, 2)

end
linkaxes!(ax...)
hidexdecorations!(ax3; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
hidexdecorations!(ax1; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
hideydecorations!(ax3; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
hideydecorations!(ax4; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
# axislegend(ax)
f[:, 3] = Legend(f, ax1, title = "Advection method", framevisible = false, labelsize=30, titlesize=36)
f


# f, ax, s = lines(cumsum(stats_Lin.time), linewidth = 3, label = "bilinear")
# lines!(ax, cumsum(stats_LinP.time) , linewidth = 3, label = "LinP")
# lines!(ax, cumsum(stats_MQS.time)  , linewidth = 3, label = "MQS")
# axislegend(ax)

# f, ax, s = lines((t), cumsum(stats_LinP.time) ./ cumsum(stats_Lin.time), linewidth = 3, label = "LinP")
# lines!(ax, (t), cumsum(stats_MQS.time) ./ cumsum(stats_Lin.time), linewidth = 3, label = "MQS")
# axislegend(ax)
# ax.xlabel = L"\text{time}"
# ax.ylabel = L"\text{slow down}"


# f, ax, s = scatter((stats_LinP.time) ./ (stats_Lin.time), markersize = 4, label = "LinP")
# scatter!(ax, (stats_MQS.time) ./ (stats_Lin.time), markersize = 4, label = "MQS")
# axislegend(ax)

# pxx, pyy  = particles1.coords;
# scatter( pxx.data[:], pyy.data[:], markersize = 4)

# pxx, pyy  = particles2.coords;
# scatter( pxx.data[:], pyy.data[:], markersize = 4)

# pxx, pyy  = particles3.coords;
# scatter( pxx.data[:], pyy.data[:], markersize = 4)


###### PLOT TIMINGS
f = Figure(size=(1800, 1600))
ax1 = Axis(f[1, 1], title = "Bilinear" , titlesize = 30, yticklabelsize = 24, xticklabelsize = 24)
ax2 = Axis(f[1, 2], title = "LinP",      titlesize = 30, yticklabelsize = 24, xticklabelsize = 24)
ax3 = Axis(f[1, 3], title = "MQS",       titlesize = 30, yticklabelsize = 24, xticklabelsize = 24)
ax4 = Axis(f[2, 1]                                     , yticklabelsize = 24, xticklabelsize = 24)
ax5 = Axis(f[2, 2]                                     , yticklabelsize = 24, xticklabelsize = 24)
ax6 = Axis(f[2, 3]                                     , yticklabelsize = 24, xticklabelsize = 24)

d   = data[1]
particles1, particles2, particles3, stats_Lin, stats_LinP, stats_MQS = d["particles1"], d["particles2"], d["particles3"], d["stats_Lin"], d["stats_LinP"], d["stats_MQS"];

nrows = size(particles1.index.data, 3)

count1 = reshape([count(particles1.index.data[:, :, i]) for i in 1:nrows], 40, 40);
count2 = reshape([count(particles2.index.data[:, :, i]) for i in 1:nrows], 40, 40);
count3 = reshape([count(particles3.index.data[:, :, i]) for i in 1:nrows], 40, 40);
h1 = heatmap!(ax1, count1 ./ 4, colorrange = (0, 1), colormap=:lipari)
h2 = heatmap!(ax2, count2 ./ 4, colorrange = (0, 1), colormap=:lipari)
h3 = heatmap!(ax3, count3 ./ 4, colorrange = (0, 1), colormap=:lipari)

d   = data[end]
particles1, particles2, particles3, stats_Lin, stats_LinP, stats_MQS = d["particles1"], d["particles2"], d["particles3"], d["stats_Lin"], d["stats_LinP"], d["stats_MQS"];

count1 = reshape([count(particles1.index.data[:, :, i]) for i in 1:nrows], 40, 40);
count2 = reshape([count(particles2.index.data[:, :, i]) for i in 1:nrows], 40, 40);
count3 = reshape([count(particles3.index.data[:, :, i]) for i in 1:nrows], 40, 40);
h4     = heatmap!(ax4, count1 ./ 20, colorrange = (0, 1), colormap=:lipari)
h5     = heatmap!(ax5, count2 ./ 20, colorrange = (0, 1), colormap=:lipari)
h6     = heatmap!(ax6, count3 ./ 20, colorrange = (0, 1), colormap=:lipari)

Colorbar(f[:, 4], h3, label = L"\text{Particles}(t = t_f) / \text{Particles}(t = 0)", labelsize = 24)

f
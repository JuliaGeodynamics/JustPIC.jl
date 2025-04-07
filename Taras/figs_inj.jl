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

###### PLOT INJECTED PARTICLES
f = Figure(size=(1900, 1200))
ax1 = Axis(f[1, 1], title = "4 particles per cell" , titlesize = 30, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, ylabelsize = 24, xlabel = L"\text{time step}", ylabel = L"\text{accumulated injected particles}")
ax2 = Axis(f[2, 1], title = "8 particles per cell" , titlesize = 30, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, ylabelsize = 24, xlabel = L"\text{time step}", ylabel = L"\text{accumulated injected particles}")
ax3 = Axis(f[1, 2], title = "12 particles per cell", titlesize = 30, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, ylabelsize = 24, xlabel = L"\text{time step}", ylabel = L"\text{accumulated injected particles}")
ax4 = Axis(f[2, 2], title = "16 particles per cell", titlesize = 30, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, ylabelsize = 24, xlabel = L"\text{time step}", ylabel = L"\text{accumulated injected particles}")
ax5 = Axis(f[1, 3], title = "20 particles per cell", titlesize = 30, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, ylabelsize = 24, xlabel = L"\text{time step}", ylabel = L"\text{accumulated injected particles}")
ax6 = Axis(f[2, 3], title = "24 particles per cell", titlesize = 30, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, ylabelsize = 24, xlabel = L"\text{time step}", ylabel = L"\text{accumulated injected particles}")

ax = ax1, ax2, ax3, ax4, ax5, ax6
for (i, ax) in enumerate(ax)
    d = data[i]
    particles1, particles2, particles3, stats_Lin, stats_LinP, stats_MQS = d["particles1"], d["particles2"], d["particles3"], d["stats_Lin"], d["stats_LinP"], d["stats_MQS"];

    scatter!(ax, cumsum(stats_Lin.injected), linewidth = 3, label = "Bilinear")
    scatter!(ax, cumsum(stats_LinP.injected), linewidth = 3, label = "LinP")
    scatter!(ax, cumsum(stats_MQS.injected), linewidth = 3, label = "MQS")
end
linkaxes!(ax...)
hidexdecorations!(ax3; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
hidexdecorations!(ax1; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
hideydecorations!(ax3; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
hideydecorations!(ax4; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
hideydecorations!(ax5; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
hideydecorations!(ax6; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
hidexdecorations!(ax5; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
f[:, 4] = Legend(f, ax1, title = "Advection method", framevisible = false, labelsize=30, titlesize=36)
f
save("injected_particles2D_injection.png", f)

###### PLOT NUMBER OF PARTICLES
f = Figure(size=(1900, 1200))
ax1 = Axis(f[1, 1], title = "4 particles per cell" , titlesize = 30, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, ylabelsize = 24, xlabel = L"\text{time step}", ylabel = L"\text{particles}")
ax2 = Axis(f[2, 1], title = "8 particles per cell" , titlesize = 30, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, ylabelsize = 24, xlabel = L"\text{time step}", ylabel = L"\text{particles}")
ax3 = Axis(f[1, 2], title = "12 particles per cell", titlesize = 30, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, ylabelsize = 24, xlabel = L"\text{time step}", ylabel = L"\text{particles}")
ax4 = Axis(f[2, 2], title = "16 particles per cell", titlesize = 30, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, ylabelsize = 24, xlabel = L"\text{time step}", ylabel = L"\text{particles}")
ax5 = Axis(f[1, 3], title = "20 particles per cell", titlesize = 30, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, ylabelsize = 24, xlabel = L"\text{time step}", ylabel = L"\text{particles}")
ax6 = Axis(f[2, 3], title = "24 particles per cell", titlesize = 30, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, ylabelsize = 24, xlabel = L"\text{time step}", ylabel = L"\text{particles}")

ax = ax1, ax2, ax3, ax4, ax5, ax6
for (i, ax) in enumerate(ax)
    d = data[i]
    particles1, particles2, particles3, stats_Lin, stats_LinP, stats_MQS = d["particles1"], d["particles2"], d["particles3"], d["stats_Lin"], d["stats_LinP"], d["stats_MQS"];

    scatter!(ax, stats_Lin.np .|> log10, linewidth = 3, label = "Bilinear")
    scatter!(ax, stats_LinP.np .|> log10, linewidth = 3, label = "LinP")
    scatter!(ax, stats_MQS.np .|> log10, linewidth = 3, label = "MQS")
end
linkaxes!(ax...)
hidexdecorations!(ax3; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
hidexdecorations!(ax1; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
hideydecorations!(ax3; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
hideydecorations!(ax4; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
hideydecorations!(ax5; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
hideydecorations!(ax6; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
hidexdecorations!(ax5; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
f[:, 4] = Legend(f, ax1, title = "Advection method", framevisible = false, labelsize=30, titlesize=36)
f
save("total_particles2D_injection.png", f)

###### PLOT TIMINGS
f = Figure(size=(1900, 1200))
ax1 = Axis(f[1, 1], title = "4 particles per cell" , titlesize = 30, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, ylabelsize = 24, xlabel = L"\text{time step}", ylabel = L"\text{accumualted run time}")
ax2 = Axis(f[2, 1], title = "8 particles per cell" , titlesize = 30, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, ylabelsize = 24, xlabel = L"\text{time step}", ylabel = L"\text{accumualted run time}")
ax3 = Axis(f[1, 2], title = "12 particles per cell", titlesize = 30, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, ylabelsize = 24, xlabel = L"\text{time step}", ylabel = L"\text{accumualted run time}")
ax4 = Axis(f[2, 2], title = "16 particles per cell", titlesize = 30, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, ylabelsize = 24, xlabel = L"\text{time step}", ylabel = L"\text{accumualted run time}")
ax5 = Axis(f[1, 3], title = "20 particles per cell", titlesize = 30, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, ylabelsize = 24, xlabel = L"\text{time step}", ylabel = L"\text{accumualted run time}")
ax6 = Axis(f[2, 3], title = "24 particles per cell", titlesize = 30, yticklabelsize = 24, xticklabelsize = 24, xlabelsize = 24, ylabelsize = 24, xlabel = L"\text{time step}", ylabel = L"\text{accumualted run time}")

ax = ax1, ax2, ax3, ax4, ax5, ax6
for (i, ax) in enumerate(ax)
    d = data[i]
    particles1, particles2, particles3, stats_Lin, stats_LinP, stats_MQS = d["particles1"], d["particles2"], d["particles3"], d["stats_Lin"], d["stats_LinP"], d["stats_MQS"];

    lines!(ax, cumsum(stats_Lin.time) ,linewidth = 3, label = "Bilinear")
    lines!(ax, cumsum(stats_LinP.time),linewidth = 3, label = "LinP")
    lines!(ax, cumsum(stats_MQS.time ),linewidth = 3, label = "MQS")
    # ylims!(ax, 0.8, 2)

    # scatter!(ax, stats_Lin.time  |> cumsum , label = "Bilinear")
    # lines!(ax, cumsum(stats_LinP.time )./ cumsum(stats_Lin.time) , color = :black, linewidth = 3, linestyle = :solid, label = "LinP")
    # lines!(ax, cumsum(stats_MQS.time  )./ cumsum(stats_Lin.time) , color = :black, linewidth = 3, linestyle = :dash, label = "MQS")
    # ylims!(ax, 0.8, 2)

end
linkaxes!(ax...)
hidexdecorations!(ax3; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
hidexdecorations!(ax1; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
hideydecorations!(ax3; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
hideydecorations!(ax4; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
hideydecorations!(ax5; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
hideydecorations!(ax6; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
hidexdecorations!(ax5; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
f[:, 4] = Legend(f, ax1, title = "Advection method", framevisible = false, labelsize=30, titlesize=36)
f
save("runtime2D_injection.png", f)

###### PLOT TIMINGS
f = Figure(size=(1800, 1600))
ax1 = Axis(f[1, 1], subtitle = L"t=0 \rightarrow \text{4 particles}" , subtitlesize = 26, title = "Bilinear" , titlesize = 30, yticklabelsize = 24, xticklabelsize = 24)
ax2 = Axis(f[1, 2], subtitle = L"t=0 \rightarrow \text{4 particles}" , subtitlesize = 26, title = "LinP",      titlesize = 30, yticklabelsize = 24, xticklabelsize = 24)
ax3 = Axis(f[1, 3], subtitle = L"t=0 \rightarrow \text{4 particles}" , subtitlesize = 26, title = "MQS",       titlesize = 30, yticklabelsize = 24, xticklabelsize = 24)
ax4 = Axis(f[2, 1], subtitle = L"t=0 \rightarrow \text{24 particles}", subtitlesize = 26, yticklabelsize = 24, xticklabelsize = 24)
ax5 = Axis(f[2, 2], subtitle = L"t=0 \rightarrow \text{24 particles}", subtitlesize = 26, yticklabelsize = 24, xticklabelsize = 24)
ax6 = Axis(f[2, 3], subtitle = L"t=0 \rightarrow \text{24 particles}", subtitlesize = 26, yticklabelsize = 24, xticklabelsize = 24)
crange = (1, 100)
d   = data[1]
particles1, particles2, particles3, stats_Lin, stats_LinP, stats_MQS = d["particles1"], d["particles2"], d["particles3"], d["stats_Lin"], d["stats_LinP"], d["stats_MQS"];

nrows = size(particles1.index.data, 3)

count1 = reshape([count(particles1.index.data[:, :, i]) for i in 1:nrows], 40, 40);
count2 = reshape([count(particles2.index.data[:, :, i]) for i in 1:nrows], 40, 40);
count3 = reshape([count(particles3.index.data[:, :, i]) for i in 1:nrows], 40, 40);
h1 = heatmap!(ax1, count1, colorrange = crange, colormap=:oleron)
h2 = heatmap!(ax2, count2, colorrange = crange, colormap=:oleron)
h3 = heatmap!(ax3, count3, colorrange = crange, colormap=:oleron)

d   = data[end]
particles1, particles2, particles3, stats_Lin, stats_LinP, stats_MQS = d["particles1"], d["particles2"], d["particles3"], d["stats_Lin"], d["stats_LinP"], d["stats_MQS"];

count1 = reshape([count(particles1.index.data[:, :, i]) for i in 1:nrows], 40, 40);
count2 = reshape([count(particles2.index.data[:, :, i]) for i in 1:nrows], 40, 40);
count3 = reshape([count(particles3.index.data[:, :, i]) for i in 1:nrows], 40, 40);
h4     = heatmap!(ax4, count1, colorrange = crange, colormap=:oleron)
h5     = heatmap!(ax5, count2, colorrange = crange, colormap=:oleron)
h6     = heatmap!(ax6, count3, colorrange = crange, colormap=:oleron)

Colorbar(f[:, 4], h3, label = L"\text{particles density}", labelsize = 24)
f
save("density2D_injection.png", f)

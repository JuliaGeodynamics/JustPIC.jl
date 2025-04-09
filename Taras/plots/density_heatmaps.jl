using JustPIC, JustPIC._2D
include("load.jl")

function density_heatmap(data)
    f = Figure(size=(1800, 1600))
    ax1 = Axis(f[1, 1], subtitlesize = 24, subtitle = L"t=0 \rightarrow 4 \text{particles per cell}", title = "Bilinear" , titlesize = 30, yticklabelsize = 24, xticklabelsize = 24)
    ax2 = Axis(f[1, 2], subtitlesize = 24, subtitle = L"t=0 \rightarrow 4 \text{particles per cell}", title = "LinP",      titlesize = 30, yticklabelsize = 24, xticklabelsize = 24)
    ax3 = Axis(f[1, 3], subtitlesize = 24, subtitle = L"t=0 \rightarrow 4 \text{particles per cell}", title = "MQS",       titlesize = 30, yticklabelsize = 24, xticklabelsize = 24)
    ax4 = Axis(f[2, 1], subtitlesize = 24, subtitle = L"t=0 \rightarrow 24 \text{particles per cell}"                                    , yticklabelsize = 24, xticklabelsize = 24)
    ax5 = Axis(f[2, 2], subtitlesize = 24, subtitle = L"t=0 \rightarrow 24 \text{particles per cell}"                                    , yticklabelsize = 24, xticklabelsize = 24)
    ax6 = Axis(f[2, 3], subtitlesize = 24, subtitle = L"t=0 \rightarrow 24 \text{particles per cell}"                                    , yticklabelsize = 24, xticklabelsize = 24)

    crange = (0, 80)

    d   = data[1]
    particles1, particles2, particles3, stats_Lin, stats_LinP, stats_MQS = d["particles1"], d["particles2"], d["particles3"], d["stats_Lin"], d["stats_LinP"], d["stats_MQS"];

    nrows = size(particles1.index.data, 3)

    count1 = reshape([count(particles1.index.data[:, :, i]) for i in 1:nrows], 40, 40);
    count2 = reshape([count(particles2.index.data[:, :, i]) for i in 1:nrows], 40, 40);
    count3 = reshape([count(particles3.index.data[:, :, i]) for i in 1:nrows], 40, 40);
    h1 = heatmap!(ax1, count1 ./ 1, colorrange = crange, colormap=:lipari)
    h2 = heatmap!(ax2, count2 ./ 1, colorrange = crange, colormap=:lipari)
    h3 = heatmap!(ax3, count3 ./ 1, colorrange = crange, colormap=:lipari)

    d   = data[end]
    particles1, particles2, particles3, stats_Lin, stats_LinP, stats_MQS = d["particles1"], d["particles2"], d["particles3"], d["stats_Lin"], d["stats_LinP"], d["stats_MQS"];

    count1 = reshape([count(particles1.index.data[:, :, i]) for i in 1:nrows], 40, 40);
    count2 = reshape([count(particles2.index.data[:, :, i]) for i in 1:nrows], 40, 40);
    count3 = reshape([count(particles3.index.data[:, :, i]) for i in 1:nrows], 40, 40);
    h4     = heatmap!(ax4, count1 ./ 1, colorrange = crange, colormap=:lipari)
    h5     = heatmap!(ax5, count2 ./ 1, colorrange = crange, colormap=:lipari)
    h6     = heatmap!(ax6, count3 ./ 1, colorrange = crange, colormap=:lipari)

    Colorbar(f[:, 4], h3, label = L"\text{Particles}", labelsize = 24)

    f

end

function plot_particles(data)
    f = Figure(size=(1800, 1600))
    ax1 = Axis(f[1, 1], subtitlesize = 24, subtitle = L"t=0 \rightarrow 4 \text{particles per cell}", title = "Bilinear" , titlesize = 30, yticklabelsize = 24, xticklabelsize = 24)
    ax2 = Axis(f[1, 2], subtitlesize = 24, subtitle = L"t=0 \rightarrow 4 \text{particles per cell}", title = "LinP",      titlesize = 30, yticklabelsize = 24, xticklabelsize = 24)
    ax3 = Axis(f[1, 3], subtitlesize = 24, subtitle = L"t=0 \rightarrow 4 \text{particles per cell}", title = "MQS",       titlesize = 30, yticklabelsize = 24, xticklabelsize = 24)
    ax4 = Axis(f[2, 1], subtitlesize = 24, subtitle = L"t=0 \rightarrow 24 \text{particles per cell}"                                    , yticklabelsize = 24, xticklabelsize = 24)
    ax5 = Axis(f[2, 2], subtitlesize = 24, subtitle = L"t=0 \rightarrow 24 \text{particles per cell}"                                    , yticklabelsize = 24, xticklabelsize = 24)
    ax6 = Axis(f[2, 3], subtitlesize = 24, subtitle = L"t=0 \rightarrow 24 \text{particles per cell}"                                    , yticklabelsize = 24, xticklabelsize = 24)


    d   = data[1]
    particles1, particles2, particles3, stats_Lin, stats_LinP, stats_MQS = d["particles1"], d["particles2"], d["particles3"], d["stats_Lin"], d["stats_LinP"], d["stats_MQS"];

    nrows = size(particles1.index.data, 3)

    px1 = particles1.coords[1].data[:]
    py1 = particles1.coords[2].data[:]
    px2 = particles2.coords[1].data[:]
    py2 = particles2.coords[2].data[:]
    px3 = particles3.coords[1].data[:]
    py3 = particles3.coords[2].data[:]

    scatter!(ax1, px1, py1, color = :black, markersize = 4)
    scatter!(ax2, px2, py2, color = :black, markersize = 4)
    scatter!(ax3, px3, py3, color = :black, markersize = 4)

    d   = data[end]
    particles1, particles2, particles3, stats_Lin, stats_LinP, stats_MQS = d["particles1"], d["particles2"], d["particles3"], d["stats_Lin"], d["stats_LinP"], d["stats_MQS"];

    px1 = particles1.coords[1].data[:]
    py1 = particles1.coords[2].data[:]
    px2 = particles2.coords[1].data[:]
    py2 = particles2.coords[2].data[:]
    px3 = particles3.coords[1].data[:]
    py3 = particles3.coords[2].data[:]

    scatter!(ax4, px1, py1, color = :black, markersize = 4)
    scatter!(ax5, px2, py2, color = :black, markersize = 4)
    scatter!(ax6, px3, py3, color = :black, markersize = 4)

    f

end


Rk4_injection = filter(
    x -> contains(x, "injection") && contains(x, "RK4.jld2"), 
    readdir(pth, join=true)
)
Rk4_injection, Rk4_injection_nps = sort_data(Rk4_injection)

Rk2_injection = filter(
    x -> contains(x, "injection") && contains(x, "RK2.jld2"), 
    readdir(pth, join=true)
)
Rk2_injection, Rk2_injection_nps = sort_data(Rk2_injection)

Rk4 = filter(
    x -> contains(x, "NO") && contains(x, "RK4.jld2"), 
    readdir(pth, join=true)
)
Rk4, Rk4_nps = sort_data(Rk4)

Rk2 = filter(
    x -> contains(x, "NO") && contains(x, "RK2.jld2"), 
    readdir(pth, join=true)
)
Rk2, Rk2_nps = sort_data(Rk2)


Rk4 = filter(
    x -> contains(x, "NO") && contains(x, "RK4.jld2"), 
    readdir("/home/albert/Documents/DevPkg/JustPIC.jl/Taras/", join=true)
)
Rk4, Rk4_nps = sort_data(Rk4)

Rk2 = filter(
    x -> contains(x, "NO") && contains(x, "RK2.jld2"), 
    readdir("/home/albert/Documents/DevPkg/JustPIC.jl/Taras/", join=true)
)
Rk2, Rk2_nps = sort_data(Rk2)


f1 = density_heatmap(Rk4_injection)
f2 = density_heatmap(Rk2_injection)
f3 = density_heatmap(Rk4)
f4 = density_heatmap(Rk2)


f1 = plot_particles(Rk4_injection)
f2 = plot_particles(Rk2_injection)
f3 = plot_particles(Rk4)
f4 = plot_particles(Rk2)

save(joinpath(@__DIR__, "particles_RK4_injection.png"), f1)
save(joinpath(@__DIR__, "particles_RK2_injection.png"), f2)
save(joinpath(@__DIR__, "particles_RK4.png"), f3)
save(joinpath(@__DIR__, "particles_RK2.png"), f4)
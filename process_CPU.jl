using CSV, DataFrames, GLMakie, Statistics

#### FUNCTIONS 

filter_by_resolution(files, res) = filter(x -> contains(x, "_n$res")  , files)
filter_by_threads(files, nt)     = filter(x -> contains(x, "_$nt.csv"), files)
filter_by_method(files, method)  = filter(x -> contains(x, method), files)

function sort_by_resolution(files)
    perm = [get_resolution(fi) for fi in files] |> sortperm
    return files[perm]
end

function get_resolution(f)
    str = split(f, "\\")[end] 
    str = split(str, "_")[3]
    resolution = parse(Int, str[2:end])
    return resolution
end

read_csv(files)         = [CSV.read(f, DataFrame) for f in files]
get_field(files, field) = [f[!, field][5:end] for f in files]

function get_field_by_method_threads(files, field, method, nt)

    # get files by method
    f_mqs_nt1    = filter_by_threads(filter_by_method(files, method), nt)
    # sort them by resolution
    sorted_files = sort_by_resolution(f_mqs_nt1)
    # read CSV files
    csvs         = read_csv(sorted_files)
    # model resolutions
    res          = [get_resolution(f) for f in sorted_files]
    # get fields
    ttotal       = get_field(csvs, field)

    return res, ttotal 
end

function plot_pie_chart(files, method)
    # get files by method
    f_mqs_nt1    = filter_by_threads(filter_by_method(files, method), 1)
    # sort them by resolution
    sorted_files = sort_by_resolution(f_mqs_nt1)
    # read CSV files
    csvs         = read_csv(sorted_files)
    # model resolutions
    res          = [get_resolution(f) for f in sorted_files]
    # get fields
    advection = get_field(csvs, "advection")
    move      = get_field(csvs, "move")
    inject    = get_field(csvs, "inject")
    p2g       = get_field(csvs, "p2g")
    g2p       = get_field(csvs, "g2p")

    # data   = [36, 12, 68, 5, 42, 27]
    colors = [:orange, :red, :blue, :purple, :green]
    data   = getindex.([advection, move, inject, p2g, g2p], 1) .|> mean
    leg    = ["advection", "move", "inject", "particle -> grid", "grid -> particle"]

    fig, ax, plt = pie(
        data,
        color        = colors,
        radius       = 4,
        inner_radius = 2,
        strokecolor  = :white,
        strokewidth  = 2,
        axis = (autolimitaspect = 1, )
    )
    ax.title = method
    hidedecorations!(ax)
    hidespines!(ax)
    Legend(fig[1,2], [PolyElement(color=c) for c in colors], leg, framevisible=false)

    fig
end

#########

fpath   = "C:\\Users\\albert\\Documents\\JP_perf\\perf"
files   = readdir(fpath, join=true)

flinear = filter(x -> contains(x, "linear") , files)
flinP   = filter(x -> contains(x, "LinP") , files)
fMOQS   = filter(x -> contains(x, "MQS") , files)

res, ttotal_mqs    = get_field_by_method_threads(files, "ttotal", "linear", 1)
res, ttotal_linear = get_field_by_method_threads(files, "ttotal", "LinP", 1)
res, ttotal_linP   = get_field_by_method_threads(files, "ttotal", "MQS", 1)

# --- plot --- # 
fig = Figure(size = (800, 800))

# plot Linear
ax = Axis(fig[1, 1], title ="MQS")
for (i, n) in enumerate(res)
    lines!(ax, 1:length(ttotal[1]), ttotal_linear[i], label = "n = $n", linewidth = 3)
    # lines!(ax, 1:length(ttotal[1]), ttotal[i] ./ np[i], label = "n = $n")
end
axislegend(ax)
fig

# # plot LinP
# ax = Axis(fig[2, 1], title ="MQS")
# for (i, n) in enumerate(res)
#     lines!(ax, 1:length(ttotal[1]), ttotal_linP[i], label = "n = $n", linewidth = 3)
#     # lines!(ax, 1:length(ttotal[1]), ttotal[i] ./ np[i], label = "n = $n")
# end
# axislegend(ax)
# fig

# # plot MQS
# ax = Axis(fig[3, 1], title ="MQS")
# for (i, n) in enumerate(res)
#     lines!(ax, 1:length(ttotal[1]), ttotal_mqs[i], label = "n = $n", linewidth = 3)
#     # lines!(ax, 1:length(ttotal[1]), ttotal[i] ./ np[i], label = "n = $n")
# end
# axislegend(ax)
# fig

###
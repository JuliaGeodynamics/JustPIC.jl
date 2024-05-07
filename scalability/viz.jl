using GLMakie, ColorSchemes
using CSV, DataFrames

getfiles(path) = filter(x -> occursin(".csv", x), readdir(path))

function nthreads(file::String)
    left  = findlast("_", file)[1] + 1
    right = findlast(".", file)[1] - 1
    parse(Int, file[left:right])
end

nthreads(files::Vector{String}) = nthreads.(files)

function ndims(file::String)
    left  = findfirst("_", file)[1] + 1
    right = findlast("x", file)[1] - 1
    parse(Int, file[left:right])
end

ndims(files::Vector{String}) = ndims.(files)

sort_dims(files) = files[sortperm(ndims(files))]
sort_threads(files) = files[sortperm(nthreads(files))]

extract_dims(files, ndim) = filter(x -> occursin("$(ndim)x$(ndim)", x), files)
extract_threads(files, nt) = filter(x -> occursin("nt_$(nt)", x), files)

function read_data(file; path ="scalability/2D/data/")
    data    = CSV.read(joinpath(path, file), DataFrame)
    # fnames  = names(data)
    delete!(data, [1])
    numrows = nrow(data)
    delete!(data, 2:numrows-1)
    data
end

function mean_data(file; path ="scalability/2D/data/")
    data    = CSV.read(joinpath(path, file), DataFrame)
    mean.(eachcol(data))
end

####

function strong_scaling(path)
    dims = 2 .^(4:10)
    cmap = cgrad(ColorSchemes.colorschemes[:thermal], length(dims), categorical = true);

    files     = getfiles(path)
    nt        = nthreads(files_dim) .>>> 1
    nt[1]     = 1

    fig  = Figure(size = (800, 600))
    ax1  = Axis(fig[1, 1], xlabel = "cores", ylabel = "speed up", title = "First time step", xticks = nt) #, xscale=log10)

    for (i, ni) in enumerate(dims) 
        files_dim = extract_dims(files, ni) |> sort_threads
        data      = mean_data.(files_dim)
        t0        = data[1][1] - data[1][4] # reference time
        tall      = [d[1] - d[4] for d in data] # reference time

        # first time step
        lines!(  ax1, nt, t0 ./ tall, color = cmap[i], label = "$(ni)x$(ni)")
        scatter!(ax1, nt, t0 ./ tall, color = cmap[i])
    end

    axislegend(ax1, position = :lt)
    fig
end

path = "scalability/2D/data/"

strong_scaling(path)
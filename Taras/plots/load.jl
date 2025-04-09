using GLMakie, JLD2

function num_part(file)
    left  = findfirst("_", file)[1]
    right = findfirst("p", file)[1]
    np = parse(Int64, file[left+1:right-1])
end


function sort_data(files)
    nps   = num_part.(files)
    isort = sortperm(nps)
    files = files[isort]
    nps   = nps[isort]
    data  = jldopen.(files);

    return data, nps
end

const pth = "/home/albert/Documents/DevPkg/JustPIC.jl/Taras/data"
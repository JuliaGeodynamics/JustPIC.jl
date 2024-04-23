using Documenter, JustPIC
push!(LOAD_PATH, "../src/")

@info "Making documentation..."
makedocs(;
    sitename="JustPIC.jl",
    authors="Albert de Montserrat and contributors",
    modules=[JustPIC, JustPIC._2D, JustPIC._3D],
    format=Documenter.HTML(; prettyurls=get(ENV, "CI", nothing) == "true"), # easier local build
    pages=[
        "Home" => "index.md",
        "Examples" => [
            "field_advection2D.md",
            "API.md",
        ]
    ],
)

deploydocs(; repo="https://github.com/JuliaGeodynamics/JustPIC.jl", devbranch="main")

pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))
using Documenter, JustPIC

@info "Making documentation..."
makedocs(;
    sitename = "JustPIC.jl",
    authors = "Albert de Montserrat and contributors",
    modules = [JustPIC],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"), # easier local build
    warnonly = Documenter.except(:footnote),
    checkdocs = :exports,
    pages = [
        "Home" => "index.md",
        "Particles" => "particles.md",
        "CellArrays" => "CellArrays.md",
        "Interpolations" => [
            "interpolations.md",
            "velocity_interpolation.md",
        ],
        "Marker chain" => "marker_chain.md",
        "Marker surface" => "marker_surface.md",
        "Examples" => [
            "field_advection2D.md",
            "field_advection2D_MPI.md",
            "field_advection3D.md",
        ],
        "I/O" => "IO.md",
        "Mixed GPU/CPU" => "mixed_CPU_GPU.md",
        "Public API" => "API.md",
    ],
)

deploydocs(; repo = "github.com/JuliaGeodynamics/JustPIC.jl", devbranch = "main")

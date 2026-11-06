using JustPIC, CairoMakie
import GLMakie
using LinearAlgebra, TOML, SHA, JLD2

const FIGURE_DIR = joinpath(@__DIR__, "figures")

vortex_velocity(x, y) = (
    sinpi(x)^2 * sinpi(2y),
    -sinpi(y)^2 * sinpi(2x),
)

deformation_velocity(x, y, z) = (
    2 * sinpi(x)^2 * sinpi(2y) * sinpi(2z),
    -sinpi(y)^2 * sinpi(2x) * sinpi(2z),
    -sinpi(z)^2 * sinpi(2x) * sinpi(2y),
)

function particle_grid(::Type{T}, n, dim; slots = 64, per_axis = 2) where {T}
    vertices = range(zero(T), one(T); length = n + 1)
    h = inv(T(n))
    centers = range(h / 2, one(T) - h / 2; length = n)
    extended = range(first(centers) - h, last(centers) + h; length = n + 2)
    velocity_grids = ntuple(d -> ntuple(i -> i == d ? vertices : extended, dim), dim)
    particles = init_particles(
        JustPIC.CPU, ntuple(_ -> per_axis, dim), slots, 0, velocity_grids...
    )
    return particles, velocity_grids
end

function vortex_case(::Type{T}; n = 160, steps = 960) where {T}
    @assert iseven(steps)
    particles, grids = particle_grid(T, n, 2; per_axis = 4)
    pi_t = T(pi)
    velocity = (
        [vortex_velocity(x, y)[1] for x in grids[1][1], y in grids[1][2]],
        [vortex_velocity(x, y)[2] for x in grids[2][1], y in grids[2][2]],
    )
    working_velocity = map(similar, velocity)
    ring(x, y) = exp(-((hypot(x - T(0.5), y - T(0.72)) - T(0.14)) / T(0.025))^2)
    field = [ring(x, y) for x in particles.xvi[1], y in particles.xvi[2]]
    values, = init_cell_arrays(particles, Val(1))
    grid2particle!(values, field, particles)
    original = map(copy, particles.coords)
    companion = (values, original...)
    initial_count = count(particles.index.data)
    duration = T(3)
    dt = duration / steps
    snapshots = Matrix{T}[]
    function snapshot!()
        particle2grid!(field, values, particles)
        physical = copy(field[2:(end - 1), 2:(end - 1)])
        @assert all(isfinite, physical) "Empty grid support in vortex figure"
        return push!(snapshots, physical)
    end
    snapshot!()
    for step in 1:steps
        amplitude = cos(pi_t * (T(step) - T(0.5)) / steps)
        for d in 1:2
            working_velocity[d] .= amplitude .* velocity[d]
        end
        advection!(particles, RungeKutta4(), working_velocity, dt)
        move_particles!(particles, companion)
        @assert count(particles.index.data) == initial_count "Particle loss in vortex figure"
        step in (steps ÷ 2, steps) && snapshot!()
    end
    active = particles.index.data
    return_error = maximum(
        sqrt.(
            sum(
                (particles.coords[d].data[active] .- original[d].data[active]) .^ 2 for d in 1:2
            )
        )
    )
    grid_return_error = norm(snapshots[3] - snapshots[1]) / norm(snapshots[1])
    @assert return_error < T(0.002) "Vortex trajectory return check failed"
    @assert grid_return_error < T(0.02) "Vortex reconstruction return check failed"
    @assert eltype(field) == T
    return (; snapshots, n, steps, duration, initial_count, return_error, grid_return_error)
end

function deformation_case(::Type{T}; n = 48, steps = 480) where {T}
    @assert iseven(steps)
    particles, grids = particle_grid(T, n, 3)
    velocity = ntuple(3) do d
        [deformation_velocity(x, y, z)[d] for x in grids[d][1], y in grids[d][2], z in grids[d][3]]
    end
    working_velocity = map(similar, velocity)
    # Select actual lattice particles; the rendered cloud has no fitted surface.
    x, y, z = map(c -> c.data, particles.coords)
    tube_distance = sqrt.((hypot.(x .- T(0.5), y .- T(0.5)) .- T(0.25)) .^ 2 .+ (z .- T(0.5)) .^ 2)
    particles.index.data .&= tube_distance .< T(0.065)
    original = map(copy, particles.coords)
    initial_count = count(particles.index.data)
    snapshots = []
    colors = []
    function snapshot!()
        active = particles.index.data
        push!(snapshots, map(c -> copy(c.data[active]), particles.coords))
        return push!(colors, atan.(original[2].data[active] .- T(0.5), original[1].data[active] .- T(0.5)))
    end
    snapshot!()
    duration = T(2)
    dt = duration / steps
    for step in 1:steps
        amplitude = cos(T(pi) * (T(step) - T(0.5)) / steps)
        for d in 1:3
            working_velocity[d] .= amplitude .* velocity[d]
        end
        advection!(particles, RungeKutta4(), working_velocity, dt)
        move_particles!(particles, original)
        @assert count(particles.index.data) == initial_count "Particle loss in 3D deformation figure"
        step in (steps ÷ 2, steps) && snapshot!()
    end
    active = particles.index.data
    return_error = maximum(
        sqrt.(
            sum(
                (particles.coords[d].data[active] .- original[d].data[active]) .^ 2 for d in 1:3
            )
        )
    )
    @assert return_error < T(0.002) "3D deformation return check failed"
    @assert all(s -> all(c -> all(isfinite, c) && all(v -> zero(T) <= v <= one(T), c), s), snapshots)
    @assert all(c -> eltype(c.data) == T, particles.coords)
    return (; snapshots, colors, n, steps, duration, initial_count, return_error)
end

# Keep 2D text and contours vector-based; embed the GL render to retain its lighting.
function export_figure(name, fig; gl = false)
    png = joinpath(FIGURE_DIR, name * ".png")
    save(png, fig; px_per_unit = 3)
    if gl
        CairoMakie.activate!()
        image = rotr90(GLMakie.FileIO.load(png))
        pdf = Figure(; size = size(fig.scene), figure_padding = 0)
        ax = Axis(pdf[1, 1]; aspect = DataAspect())
        image!(ax, image)
        hidedecorations!(ax)
        hidespines!(ax)
        tightlimits!(ax)
        fig = pdf
    end
    save(joinpath(FIGURE_DIR, name * ".pdf"), fig)
    return save(joinpath(FIGURE_DIR, name * ".svg"), fig)
end

function draw_vortex(result)
    CairoMakie.activate!()
    fig = Figure(; size = (1200, 480), figure_padding = 28)
    Label(fig[0, 1:4], "Stretch, transport, recover"; fontsize = 26, font = :bold, halign = :left)
    titles = ("a   Initial ring · t = 0", "b   Maximum deformation · t = 1.5", "c   Reversed flow · t = 3")
    nodes = range(0, 1; length = result.n + 1)
    for i in 1:3
        ax = Axis(
            fig[1, i]; aspect = DataAspect(), title = titles[i], xlabel = "x / L", ylabel = i == 1 ? "y / L" : "",
            xticks = 0:0.25:1, yticks = 0:0.25:1
        )
        hm = heatmap!(ax, nodes, nodes, result.snapshots[i]; colormap = :batlow, colorrange = (0, 1), rasterize = 3)
        direction = i == 3 ? -1 : 1
        streamplot!(
            ax, (x, y) -> Point2f((direction .* vortex_velocity(x, y))...), 0 .. 1, 0 .. 1;
            color = _ -> RGBAf(1, 1, 1, 0.25), linewidth = 0.8,
            gridsize = (14, 14), density = 0.7, stepsize = 0.004, maxsteps = 1200,
            arrow_size = i == 2 ? 0 : 7
        )
        contour!(ax, nodes, nodes, result.snapshots[i]; levels = [0.25, 0.65], color = (:white, 0.45), linewidth = 0.7)
        xlims!(ax, 0, 1)
        ylims!(ax, 0, 1)
        hidespines!(ax, :t, :r)
        i > 1 && hideydecorations!(ax)
        i == 3 && Colorbar(fig[1, 4], hm; label = "Transported scalar", width = 12, ticks = 0:0.25:1)
    end
    Label(
        fig[2, 1:4], "2D REVERSIBLE VORTEX   /   $(result.n) × $(result.n) cells   /   $(result.initial_count) particles   /   RK4";
        fontsize = 12, color = :slategray, halign = :left
    )
    Label(
        fig[3, 1:4], "Flow curves: spatial pattern; arrows: direction before/after reversal (midpoint at rest).";
        fontsize = 11, color = :slategray, halign = :left
    )
    return export_figure("vortex_transport", fig)
end

function draw_deformation(result)
    GLMakie.activate!(; ssao = true, fxaa = true)
    fig = Figure(; size = (1200, 560), figure_padding = 28)
    Label(fig[0, 1:3], "Twist, fold, recover"; fontsize = 26, font = :bold, halign = :left)
    titles = ("a   Initial torus · t = 0", "b   Folded cloud · t = 1", "c   Recovered torus · t = 2")
    for i in 1:3
        ax = Axis3(
            fig[1, i]; title = titles[i], aspect = (1, 1, 1),
            xlabel = "x / L", ylabel = "y / L", zlabel = "z / L",
            azimuth = 1.7pi, elevation = 0.2pi, perspectiveness = 0.25,
            xticks = [0.2, 0.5, 0.8], yticks = [0.2, 0.5, 0.8], zticks = [0.2, 0.5, 0.8],
            xypanelcolor = (:slategray, 0.055), xzpanelcolor = (:slategray, 0.025), yzpanelcolor = (:slategray, 0.025),
            xgridcolor = (:slategray, 0.15), ygridcolor = (:slategray, 0.15), zgridcolor = (:slategray, 0.15),
            xspinewidth = 0.6, yspinewidth = 0.6, zspinewidth = 0.6
        )
        GLMakie.Makie.set_ambient_light!(ax.scene, RGBf(0.32, 0.34, 0.38))
        GLMakie.Makie.set_lights!(
            ax.scene, [
                DirectionalLight(RGBf(0.95, 0.91, 0.85), Vec3f(-0.6, -0.8, -1)),
                DirectionalLight(RGBf(0.28, 0.35, 0.45), Vec3f(0.8, 0.4, -0.2)),
            ]
        )
        ax.scene.ssao.radius[] = 0.035
        ax.scene.ssao.bias[] = 0.002
        ax.scene.ssao.blur[] = 3
        direction = i == 3 ? -1 : 1
        streamplot!(
            ax, (x, y, z) -> Point3f((direction .* deformation_velocity(x, y, z))...),
            0.1 .. 0.9, 0.1 .. 0.9, 0.1 .. 0.9;
            color = _ -> RGBAf(0.25, 0.35, 0.45, 0.22), linewidth = 1.0,
            gridsize = (6, 6, 6), density = 0.35, stepsize = 0.003, maxsteps = 1200,
            arrow_size = i == 2 ? 0 : 0.018, transparency = true
        )
        meshscatter!(
            ax, result.snapshots[i]...; color = result.colors[i],
            colormap = :phase, colorrange = (-pi, pi), markersize = 0.0055,
            diffuse = Vec3f(0.75), specular = Vec3f(0.65), shininess = 48,
            ssao = true, fxaa = true
        )
        limits!(ax, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9)
    end
    Colorbar(
        fig[2, 1:3]; colormap = :phase, limits = (-pi, pi), vertical = false,
        label = "Initial azimuth (rad)", ticks = ([-pi, -pi / 2, 0, pi / 2, pi], ["−π", "−π/2", "0", "π/2", "π"]),
        width = Relative(0.48), height = 10, flipaxis = false
    )
    Label(
        fig[3, 1:3], "3D REVERSIBLE DEFORMATION   /   $(result.n)³ cells   /   $(result.initial_count) particles   /   colour tracks initial azimuth";
        fontsize = 12, color = :slategray, halign = :left
    )
    Label(
        fig[4, 1:3], "Flow curves: spatial pattern; arrows: direction before/after reversal (midpoint at rest).";
        fontsize = 11, color = :slategray, halign = :left
    )
    return export_figure("deformation_3d", fig; gl = true)
end

function render_figures(vortex, deformation)
    @assert length(vortex.snapshots) == length(deformation.snapshots) == length(deformation.colors) == 3
    @assert all(s -> size(s) == (vortex.n + 1, vortex.n + 1) && all(isfinite, s), vortex.snapshots)
    @assert all(i -> all(c -> length(c) == length(deformation.colors[i]), deformation.snapshots[i]), 1:3)
    set_theme!(
        Theme(;
            font = "TeX Gyre Heros", fontsize = 14, backgroundcolor = :white, textcolor = :gray15,
            Axis = (
                titlealign = :left, titlesize = 16, titlegap = 14, xgridvisible = false, ygridvisible = false,
                xtickcolor = :gray65, ytickcolor = :gray65, spinewidth = 0.8,
            ),
            Axis3 = (titlealign = :left, titlesize = 16, titlegap = 12, xticklabelsize = 11, yticklabelsize = 11, zticklabelsize = 11),
            Colorbar = (spinewidth = 0, tickcolor = :gray65, ticklabelsize = 12)
        )
    )
    draw_vortex(vortex)
    draw_deformation(deformation)
    open(joinpath(FIGURE_DIR, "rendering.toml"), "w") do io
        TOML.print(
            io, Dict(
                "script_sha256" => bytes2hex(sha256(read(@__FILE__))),
                "docs_manifest_sha256" => bytes2hex(sha256(read(joinpath(@__DIR__, "..", "docs", "Manifest.toml")))),
                "figure_data_sha256" => bytes2hex(sha256(read(joinpath(FIGURE_DIR, "figure_data.jld2")))),
                "glmakie_version" => string(pkgversion(GLMakie)),
                "cairomakie_version" => string(pkgversion(CairoMakie)),
                "deformation_renderer" => "GLMakie: lit spheres, SSAO, FXAA, spatial streamlines; raster embedded in PDF/SVG",
                "vortex_renderer" => "CairoMakie: raster scalar field, vector contours, streamlines and typography"
            ); sorted = true
        )
    end
    return nothing
end

function main()
    mkpath(FIGURE_DIR)
    if ARGS == ["--render-only"]
        data = load(joinpath(FIGURE_DIR, "figure_data.jld2"))
        render_figures(data["vortex"], data["deformation"])
        return nothing
    end
    isempty(ARGS) || error("Usage: generate_figures.jl [--render-only]")
    # The small Float32 cases exercise the same numerical paths without claiming GPU validation.
    float32_vortex = vortex_case(Float32; n = 24, steps = 240)
    float32_deformation = deformation_case(Float32; n = 16, steps = 240)
    vortex = vortex_case(Float64)
    deformation = deformation_case(Float64)
    jldsave(joinpath(FIGURE_DIR, "figure_data.jld2"); vortex, deformation)
    metadata = Dict(
        "julia_version" => string(VERSION),
        "justpic_version" => string(pkgversion(JustPIC)),
        "justpic_source" => pathof(JustPIC),
        "cairomakie_version" => string(pkgversion(CairoMakie)),
        "git_revision" => strip(read(`git -C $(@__DIR__) rev-parse HEAD`, String)),
        "script_sha256" => bytes2hex(sha256(read(@__FILE__))),
        "docs_manifest_sha256" => bytes2hex(sha256(read(joinpath(@__DIR__, "..", "docs", "Manifest.toml")))),
        "backend" => "CPU",
        "precision" => "Float64 (plotted), Float32 (small self-checks)",
        "float32_self_checks" => Dict(
            "vortex_cells_per_axis" => float32_vortex.n,
            "vortex_steps" => float32_vortex.steps,
            "max_particle_return_error" => float32_vortex.return_error,
            "relative_grid_return_error" => float32_vortex.grid_return_error,
            "deformation_cells_per_axis" => float32_deformation.n,
            "deformation_steps" => float32_deformation.steps,
            "max_deformation_return_error" => float32_deformation.return_error
        ),
        "vortex" => Dict(
            "cells_per_axis" => vortex.n, "steps" => vortex.steps,
            "duration" => vortex.duration, "particles" => vortex.initial_count,
            "max_particle_return_error" => vortex.return_error,
            "relative_grid_return_error" => vortex.grid_return_error
        ),
        "deformation" => Dict(
            "cells_per_axis" => deformation.n, "steps" => deformation.steps,
            "duration" => deformation.duration, "particles" => deformation.initial_count,
            "max_particle_return_error" => deformation.return_error
        ),
    )
    open(joinpath(FIGURE_DIR, "validation.toml"), "w") do io
        TOML.print(io, metadata; sorted = true)
    end
    render_figures(vortex, deformation)
    println("Figures and validation metadata written to ", FIGURE_DIR)
    return metadata
end

main()

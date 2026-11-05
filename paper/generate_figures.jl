using JustPIC, CairoMakie
using LinearAlgebra, TOML, SHA, JLD2

const FIGURE_DIR = joinpath(@__DIR__, "figures")

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
        [sinpi(x)^2 * sinpi(2y) for x in grids[1][1], y in grids[1][2]],
        [-sinpi(y)^2 * sinpi(2x) for x in grids[2][1], y in grids[2][2]],
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

function torus_case(::Type{T}; n = 40, steps = 120) where {T}
    @assert iseven(steps)
    particles, grids = particle_grid(T, n, 3)
    axis = T[1, 0.4, 0.7]
    axis ./= norm(axis)
    omega = Tuple(axis)
    velocity_at(x, y, z) = (
        omega[2] * (z - T(0.5)) - omega[3] * (y - T(0.5)),
        omega[3] * (x - T(0.5)) - omega[1] * (z - T(0.5)),
        omega[1] * (y - T(0.5)) - omega[2] * (x - T(0.5)),
    )
    velocity = ntuple(3) do d
        [velocity_at(x, y, z)[d] for x in grids[d][1], y in grids[d][2], z in grids[d][3]]
    end
    # Select a torus from a regular particle lattice; no injection or surface fitting.
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
    angle = T(1.4)
    dt = angle / steps
    for step in 1:steps
        advection!(particles, RungeKutta4(), velocity, dt)
        move_particles!(particles, original)
        @assert count(particles.index.data) == initial_count "Particle loss in torus figure"
        step in (steps ÷ 2, steps) && snapshot!()
    end
    active = particles.index.data
    r0 = reduce(hcat, (c.data[active] .- T(0.5) for c in original))
    cross_terms = hcat(
        omega[2] .* r0[:, 3] .- omega[3] .* r0[:, 2],
        omega[3] .* r0[:, 1] .- omega[1] .* r0[:, 3],
        omega[1] .* r0[:, 2] .- omega[2] .* r0[:, 1],
    )
    exact = cos(angle) .* r0 .+ sin(angle) .* cross_terms .+
        (one(T) - cos(angle)) .* (r0 * axis) * axis'
    current = reduce(hcat, (c.data[active] .- T(0.5) for c in particles.coords))
    rotation_error = maximum(sqrt.(sum(abs2, current - exact; dims = 2)))
    @assert rotation_error < T(0.0001) "Analytical 3D rotation check failed"
    @assert all(c -> eltype(c.data) == T, particles.coords)
    return (; snapshots, colors, n, steps, angle, initial_count, rotation_error)
end

function export_figure(name, fig)
    save(joinpath(FIGURE_DIR, name * ".png"), fig; px_per_unit = 3)
    return save(joinpath(FIGURE_DIR, name * ".pdf"), fig)
end

function draw_vortex(result)
    fig = Figure(; size = (1080, 430), fontsize = 14, backgroundcolor = :white)
    Label(fig[0, 1:3], "Stretch, transport, recover"; fontsize = 24, font = :bold, halign = :left)
    titles = ("a   Initial ring · t = 0", "b   Maximum deformation · t = 1.5", "c   Reversed flow · t = 3")
    nodes = range(0, 1; length = result.n + 1)
    for i in 1:3
        ax = Axis(fig[1, i]; aspect = DataAspect(), title = titles[i], xlabel = "x / L", ylabel = i == 1 ? "y / L" : "")
        hm = heatmap!(ax, nodes, nodes, result.snapshots[i]; colormap = :batlow, colorrange = (0, 1))
        xlims!(ax, 0, 1)
        ylims!(ax, 0, 1)
        i > 1 && hideydecorations!(ax; grid = false)
        i == 3 && Colorbar(fig[1, 4], hm; label = "Transported scalar", width = 12)
    end
    Label(
        fig[2, 1:4], "2D reversible vortex  /  $(result.n) × $(result.n) cells  /  $(result.initial_count) particles  /  RK4";
        fontsize = 13, color = :gray35, halign = :left
    )
    return export_figure("vortex_transport", fig)
end

function draw_torus(result)
    fig = Figure(; size = (1080, 430), fontsize = 14, backgroundcolor = :white)
    Label(fig[0, 1:3], "One particle cloud, three orientations"; fontsize = 24, font = :bold, halign = :left)
    titles = ("a   Initial torus", "b   Rotation · θ = 0.7 rad", "c   Rotation · θ = 1.4 rad")
    for i in 1:3
        ax = Axis3(
            fig[1, i]; title = titles[i], aspect = (1, 1, 1),
            xlabel = "x / L", ylabel = "y / L", zlabel = "z / L",
            azimuth = 1.7pi, elevation = 0.25pi,
            xticks = [0.2, 0.5, 0.8], yticks = [0.2, 0.5, 0.8], zticks = [0.2, 0.5, 0.8]
        )
        scatter!(
            ax, result.snapshots[i]...; color = result.colors[i],
            colormap = :twilight, colorrange = (-pi, pi), markersize = 2.4
        )
        limits!(ax, 0.15, 0.85, 0.15, 0.85, 0.15, 0.85)
    end
    Colorbar(
        fig[2, 1:3]; colormap = :twilight, limits = (-pi, pi), vertical = false,
        label = "Initial azimuth (rad)", ticks = ([-pi, 0, pi], ["−π", "0", "π"]),
        width = Relative(0.45), height = 10
    )
    Label(
        fig[3, 1:3], "3D solid-body rotation  /  $(result.n)³ cells  /  $(result.initial_count) particles  /  colour tracks initial azimuth";
        fontsize = 13, color = :gray35, halign = :left
    )
    return export_figure("torus_rotation", fig)
end

function main()
    mkpath(FIGURE_DIR)
    set_theme!(Theme(; font = "TeX Gyre Heros", Axis = (titlealign = :left, titlesize = 16)))
    # The small Float32 cases exercise the same numerical paths without claiming GPU validation.
    float32_vortex = vortex_case(Float32; n = 24, steps = 240)
    float32_torus = torus_case(Float32; n = 16, steps = 60)
    vortex = vortex_case(Float64)
    torus = torus_case(Float64)
    draw_vortex(vortex)
    draw_torus(torus)
    jldsave(joinpath(FIGURE_DIR, "figure_data.jld2"); vortex, torus)
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
            "torus_cells_per_axis" => float32_torus.n,
            "torus_steps" => float32_torus.steps,
            "max_analytical_rotation_error" => float32_torus.rotation_error
        ),
        "vortex" => Dict(
            "cells_per_axis" => vortex.n, "steps" => vortex.steps,
            "duration" => vortex.duration, "particles" => vortex.initial_count,
            "max_particle_return_error" => vortex.return_error,
            "relative_grid_return_error" => vortex.grid_return_error
        ),
        "torus" => Dict(
            "cells_per_axis" => torus.n, "steps" => torus.steps,
            "angle_radians" => torus.angle, "particles" => torus.initial_count,
            "max_analytical_rotation_error" => torus.rotation_error
        ),
    )
    open(joinpath(FIGURE_DIR, "validation.toml"), "w") do io
        TOML.print(io, metadata; sorted = true)
    end
    println("Figures and validation metadata written to ", FIGURE_DIR)
    return metadata
end

main()

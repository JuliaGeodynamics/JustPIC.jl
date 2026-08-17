using CairoMakie, JLD2, JustPIC

const backend = JustPIC.CPU
const MATTERHORN_EXTENT_METERS = (1542.7, 2212.2)

function extend_centers(xv)
    xc = [(xv[i] + xv[i + 1]) / 2 for i in 1:(length(xv) - 1)]
    return [2 * xc[1] - xc[2]; xc; 2 * xc[end] - xc[end - 1]]
end

function matterhorn_topography()
    elevation = load_object(joinpath(@__DIR__, "matterhorn.jld2"))
    zmin, zmax = extrema(elevation)
    topo = @. 0.45 + 0.1 * (elevation - zmin) / (zmax - zmin)
    return topo, (zmin, zmax)
end

function periodic_topography(n; x0 = 0.1)
    xv = yv = LinRange(0, 1, n)
    distance_x(x) = min(abs(x - x0), 1 - abs(x - x0))
    return [0.5 + 0.05 * exp(-50 * (distance_x(x)^2 + (y - 0.5)^2)) for x in xv, y in yv]
end

function marker_surface_example(topo; periodic_1 = false, suffix = "", elevation_range = nothing, velocity_resolution = 33)
    surface_resolution = size(topo, 1)
    size(topo) == (surface_resolution, surface_resolution) || throw(ArgumentError("example topography must be square"))
    surface_xv = surface_yv = LinRange(0, 1, surface_resolution)
    xv = yv = zv = LinRange(0, 1, velocity_resolution)
    xc, yc, zc = extend_centers(xv), extend_centers(yv), extend_centers(zv)
    grid_vxi = ((xv, yc, zc), (xc, yv, zc), (xc, yc, zv))
    kx = periodic_1 ? 2π : π
    V = (
        TA(backend)([0.1 * sin(kx * x) * cos(π * z) for x in grid_vxi[1][1], y in grid_vxi[1][2], z in grid_vxi[1][3]]),
        TA(backend)(zeros(length(grid_vxi[2][1]), length(grid_vxi[2][2]), length(grid_vxi[2][3]))),
        TA(backend)([-0.1 * (kx / π) * cos(kx * x) * sin(π * z) for x in grid_vxi[3][1], y in grid_vxi[3][2], z in grid_vxi[3][3]]),
    )
    surf = init_marker_surface(backend, surface_xv, surface_yv, topo; periodic_1)
    assets = joinpath(@__DIR__, "..", "src", "assets")

    save(joinpath(assets, "initial_condition_surface$(suffix).png"), surface_figure(surf; elevation_range))
    for _ in 1:10
        advect_marker_surface!(surf, V, grid_vxi, 0.02; max_slope_angle = 0)
    end
    save(joinpath(assets, "uplifted_surface$(suffix).png"), surface_figure(surf; elevation_range))
    return nothing
end

function surface_figure(surf; elevation_range = nothing)
    x, y, topo = Array(surf.xv), Array(surf.yv), Array(surf.topo)
    if isnothing(elevation_range)
        axis_kwargs = (
            aspect = (1, 1, 0.5), xlabel = "x", ylabel = "y",
            zlabel = "surface elevation"
        )
    else
        zmin, zmax = elevation_range
        x = MATTERHORN_EXTENT_METERS[1] .* x
        y = MATTERHORN_EXTENT_METERS[2] .* y
        topo = @. zmin + (topo - 0.45) * (zmax - zmin) / 0.1
        axis_kwargs = (
            # Viewed from the northeast, the classic Zermatt profile of the Matterhorn
            aspect = :data, azimuth = 0.2π, elevation = π / 12, perspectiveness = 0.5,
            viewmode = :fit,
            xlabel = "local easting (m)", ylabel = "local northing (m)", zlabel = "elevation (m)"
        )
    end
    fig = Figure(size = (850, 650))
    ax = Axis3(fig[1, 1]; axis_kwargs...)
    colormap = isnothing(elevation_range) ? :oleron : :grayC10
    plt = surface!(ax, x, y, topo; colormap)
    Colorbar(fig[1, 2], plt; label = axis_kwargs.zlabel)
    display(fig)
    return fig
end

matterhorn, matterhorn_elevation_range = matterhorn_topography()
marker_surface_example(matterhorn; elevation_range = matterhorn_elevation_range)
marker_surface_example(periodic_topography(33); periodic_1 = true, suffix = "_periodic")

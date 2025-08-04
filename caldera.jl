# const isCUDA = false
const isCUDA = true

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO

const backend = @static if isCUDA
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustRelax.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

using ParallelStencil, ParallelStencil.FiniteDifferences2D

@static if isCUDA
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

using JustPIC, JustPIC._2D
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend_JP = @static if isCUDA
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

# Load script dependencies
# using  GLMakie, Statistics, Dates
using  Statistics, Dates

# Load file with all the rheology configurations

## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

import ParallelStencil.INDICES
const idx_k = INDICES[2]
macro all_k(A)
    return esc(:($A[$idx_k]))
end
using GeophysicalModelGenerator

function setup2D(
        nx, nz;
        sticky_air = 5.0e0,
        dimensions = (30.0e0, 20.0e0), # extent in x and y in km
        flat = true,
        chimney = false,
        layers = 1,
        volcano_size = (3.0e0, 5.0e0),
        conduit_radius = 0.2,
        chamber_T = 1.0e3,
        chamber_depth = 5.0e0,
        chamber_radius = 2.0e0,
        aspect_x = 1.5,
    )

    Lx = Ly = dimensions[1]
    x = range(0.0, Lx, nx)
    y = range(0.0, Ly, 2)
    z = range(-dimensions[2], sticky_air, nz)
    Grid = CartData(xyz_grid(x, y, z))

    # Allocate Phase and Temp arrays
    air_phase = layers + 7
    # Phases = fill(6, nx, 2, nz);
    Phases = fill(air_phase, nx, 2, nz)
    Temp = fill(0.0, nx, 2, nz)

    add_box!(
        Phases, Temp, Grid;
        xlim = (minimum(Grid.x.val), maximum(Grid.x.val)),
        ylim = (minimum(Grid.y.val), maximum(Grid.y.val)),
        zlim = (minimum(Grid.z.val), 0.0),
        phase = LithosphericPhases(Layers = [chamber_depth], Phases = [1, 2]),
        T = HalfspaceCoolingTemp(Age = 20)
    )

    add_stripes!(Phases, Grid;
        stripAxes = (0,0,1),
        phase = ConstantPhase(1),
        stripePhase = ConstantPhase(layers + 5),
        # stripeWidth=0.002,
        stripeSpacing=0.5
    )

    if !flat
        heights = layers > 1 ? [volcano_size[1] - i * (volcano_size[1] / layers) for i in 0:(layers - 1)] : volcano_size[1]
        for (i, height) in enumerate(heights)
            add_volcano!(
                Phases, Temp, Grid;
                volcanic_phase = i + 4,  # Change phase for each layer
                center = (mean(Grid.x.val), 0.0),
                height = height,
                radius = volcano_size[2],
                crater = 0.5,
                base = 0.0,
                background = nothing,
                # T               = HalfspaceCoolingTemp(Age=20)
                T = i == 1 ? HalfspaceCoolingTemp(Age = 20) : nothing,
            )
        end
    end

    add_ellipsoid!(
        Phases, Temp, Grid;
        cen = (mean(Grid.x.val), 0, -chamber_depth),
        axes = (chamber_radius * aspect_x, 2.5, chamber_radius),
        phase = ConstantPhase(3),
        T = ConstantTemp(T = chamber_T - 100.0e0)
    )

    add_ellipsoid!(
        Phases, Temp, Grid;
        cen = (mean(Grid.x.val), 0, -(chamber_depth - (chamber_radius / 2))),
        axes = ((chamber_radius / 1.25) * aspect_x, 2.5, (chamber_radius / 2)),
        phase = ConstantPhase(4),
        T = ConstantTemp(T = chamber_T)
    )

    # add_sphere!(Phases, Temp, Grid;
    #     cen    = (mean(Grid.x.val), 0, -(chamber_depth-(chamber_radius/2))),
    #     radius = (chamber_radius/2),
    #     phase  = ConstantPhase(4),
    #     T      = ConstantTemp(T=chamber_T+100)
    # )


    if chimney
        add_cylinder!(
            Phases, Temp, Grid;
            base = (mean(Grid.x.val), 0, -(chamber_depth - chamber_radius)),
            cap = (mean(Grid.x.val), 0, flat ? 0.0e0 : volcano_size[1]),
            radius = conduit_radius,
            phase = ConstantPhase(layers + 6),
            # T      = ConstantTemp(T=chamber_T),
        )
    end

    Grid = addfield(Grid, (; Phases, Temp))
    li = (abs(last(x) - first(x)), abs(last(z) - first(z))) .* 1.0e3
    origin = (x[1], z[1]) .* 1.0e3

    ph = Phases[:, 1, :]
    T = Temp[:, 1, :] .+ 273
    V_total = (4 / 3 * π * (chamber_radius * aspect_x) * chamber_radius * (chamber_radius * aspect_x)) * 1.0e9
    V_eruptible = (4 / 3 * π * (chamber_radius / 1.25) * aspect_x * (chamber_radius / 2) * ((chamber_radius / 1.25) * aspect_x)) * 1.0e9
    R = ((chamber_depth - chamber_radius)) / (chamber_radius * aspect_x)
    chamber_diameter = 2 * (chamber_radius * aspect_x)
    chamber_erupt = 2 * ((chamber_radius / 1.25) * aspect_x)
    printstyled("Magma volume of the initial chamber:$(round(ustrip.(uconvert(u"km^3", (V_total)u"m^3")); digits = 5)) km³ \n"; bold = true, color = :red, blink = true)
    printstyled("Eruptible magma volume: $(round(ustrip.(uconvert(u"km^3", (V_eruptible)u"m^3")); digits = 5)) km³ \n"; bold = true, color = :red, blink = true)
    printstyled("Roof ratio (Depth/half-axis width): $R \n"; bold = true, color = :cyan)
    printstyled("Chamber diameter: $(round(chamber_diameter; digits = 3)) km \n"; bold = true, color = :light_yellow)
    printstyled("Eruptible chamber diameter: $(round(chamber_erupt; digits = 3)) km \n"; bold = true, color = :light_yellow)
    # write_paraview(Grid, "Volcano2D")
    return li, origin, ph, T, Grid, V_total, V_eruptible, layers, air_phase
end

function copyinn_x!(A, B)
    @parallel function f_x(A, B)
        @all(A) = @inn_x(B)
        return nothing
    end

    return @parallel f_x(A, B)
end

# Initial pressure profile - not accurate
@parallel function init_P!(P, ρg, z)
    @all(P) = abs(@all(ρg) * @all_k(z)) #* <(@all_k(z), 0.0)
    return nothing
end

function apply_pure_shear(Vx, Vy, εbg, xvi, lx, ly)
    xv, yv = xvi

    @parallel_indices (i, j) function pure_shear_x!(Vx, εbg, lx)
        xi = xv[i]
        Vx[i, j + 1] = εbg * (xi - lx * 0.5)
        return nothing
    end

    @parallel_indices (i, j) function pure_shear_y!(Vy, εbg, ly)
        yi = yv[j]
        Vy[i + 1, j] = abs(yi) * εbg
        return nothing
    end

    nx, ny = size(Vx)
    @parallel (1:nx, 1:(ny - 2)) pure_shear_x!(Vx, εbg, lx)
    nx, ny = size(Vy)
    @parallel (1:(nx - 2), 1:ny) pure_shear_y!(Vy, εbg, ly)

    return nothing
end

function extract_topo_from_GMG_phases(phases_GMG, xvi, air_phase)
    topo_idx = [findfirst(x -> x == air_phase, row) - 1 for row in eachrow(phases_GMG)]
    yv = xvi[2]
    topo_y = yv[topo_idx]
    return topo_y
end

function thermal_anomaly!(Temp, Ω_T, phase_ratios, T_chamber, T_air, conduit_phase, magma_phase, anomaly_phase, air_phase)

    @parallel_indices (i, j) function _thermal_anomaly!(Temp, Ω_T, T_chamber, T_air, vertex_ratio, conduit_phase, magma_phase, anomaly_phase, air_phase)
        # quick escape
        # conduit_ratio_ij = @index vertex_ratio[conduit_phase, i, j]
        magma_ratio_ij = @index vertex_ratio[magma_phase, i, j]
        anomaly_ratio_ij = @index vertex_ratio[anomaly_phase, i, j]
        air_ratio_ij = @index vertex_ratio[air_phase, i, j]

        # if conduit_ratio_ij > 0.5 || magma_ratio_ij > 0.5
        if anomaly_ratio_ij > 0.5
            Ω_T[i + 1, j] = Temp[i + 1, j] = T_chamber
        elseif magma_ratio_ij > 0.5
            Ω_T[i + 1, j] = Temp[i + 1, j] = T_chamber - 100.0e0
        elseif air_ratio_ij > 0.5
            Ω_T[i + 1, j] = Temp[i + 1, j] = T_air
        end

        return nothing
    end

    ni = size(phase_ratios.vertex)

    @parallel (@idx ni) _thermal_anomaly!(Temp, Ω_T, T_chamber, T_air, phase_ratios.vertex, conduit_phase, magma_phase, anomaly_phase, air_phase)

    @views Ω_T[1, :] .= Ω_T[2, :]
    @views Ω_T[end, :] .= Ω_T[end - 1, :]
    @views Temp[1, :] .= Temp[2, :]
    @views Temp[end, :] .= Temp[end - 1, :]

    return nothing
end

function plot_particles(particles, pPhases, chain; clrmap = :roma)
    p = particles.coords
    # pp = [argmax(p) for p in phase_ratios.center] #if you want to plot it in a heatmap rather than scatter
    ppx, ppy = p
    pxv = ppx.data[:] ./ 1.0e3
    pyv = ppy.data[:] ./ 1.0e3
    clr = pPhases.data[:]
    chain_x = chain.coords[1].data[:] ./ 1.0e3
    chain_y = chain.coords[2].data[:] ./ 1.0e3
    idxv = particles.index.data[:]
    f, ax, h = scatter(Array(pxv[idxv]), Array(pyv[idxv]), color = Array(clr[idxv]), colormap = clrmap, markersize = 1)
    scatter!(ax, Array(chain_x), Array(chain_y), color = :red, markersize = 1)
    Colorbar(f[1, 2], h)
    return f
end

function make_it_go_boom!(Q, threshold, cells, ϕ, V_erupt, V_tot, di, phase_ratios, magma_phase, anomaly_phase)

    @parallel_indices (i, j) function _make_it_go_boom!(Q, threshold, cells, ϕ, V_erupt, V_tot, dx, dy, center_ratio, magma_phase, anomaly_phase)

        magma_ratio_ij = @index center_ratio[magma_phase, i, j]
        anomaly_ratio_ij = @index center_ratio[anomaly_phase, i, j]
        total_fraction = magma_ratio_ij + anomaly_ratio_ij
        ϕ_ij = ϕ[i, j]
        cells_ij = cells[i, j]

        if (anomaly_ratio_ij > 0.5 || magma_ratio_ij > 0.5) && ϕ_ij ≥ threshold
            Q[i, j] = (V_erupt * inv(V_tot)) * ((total_fraction * cells_ij) * inv(numcells(cells)))
        end
        return nothing
    end

    ni = size(phase_ratios.center)

    @parallel (@idx ni) _make_it_go_boom!(Q, threshold, cells, ϕ, V_erupt, V_tot, di..., phase_ratios.center, magma_phase, anomaly_phase)
    V_tot += V_erupt

    return V_tot, V_erupt
end

function make_it_go_boom!(Q, threshold, cells, ϕ, ρg, Δρ, V_erupt, V_tot, di, phase_ratios, magma_phase, anomaly_phase)

    @parallel_indices (i, j) function _make_it_go_boom!(Q, threshold, cells, ϕ, ρg, Δρ, V_erupt, V_tot, dx, dy, center_ratio, magma_phase, anomaly_phase)

        magma_ratio_ij = @index center_ratio[magma_phase, i, j]
        anomaly_ratio_ij = @index center_ratio[anomaly_phase, i, j]
        total_fraction = magma_ratio_ij + anomaly_ratio_ij
        ϕ_ij = ϕ[i, j]
        cells_ij = cells[i, j]
        ρg_ij = ρg[i, j]

        if (anomaly_ratio_ij > 0.5 || magma_ratio_ij > 0.5) && ϕ_ij ≥ threshold
            Q[i, j] =(((Δρ-ρg_ij) * inv(ρg_ij)) + (V_erupt * inv(V_tot))) * ((total_fraction * cells_ij) * inv(numcells(cells)))
        end
        return nothing
    end

    ni = size(phase_ratios.center)

    @parallel (@idx ni) _make_it_go_boom!(Q, threshold, cells, ϕ, ρg, Δρ, V_erupt, V_tot, di..., phase_ratios.center, magma_phase, anomaly_phase)
    V_tot += V_erupt

    return V_tot, V_erupt
end

function compute_cells_for_Q!(cells, threshold, phase_ratios, magma_phase, anomaly_phase, melt_fraction)
    @parallel_indices (I...) function _compute_cells_for_Q!(cells, threshold, center_ratio, magma_phase, anomaly_phase, melt_fraction)
        magma_ratio_ij = @index center_ratio[magma_phase, I...]
        anomaly_ratio_ij = @index center_ratio[anomaly_phase, I...]
        melt_fraction_ij = melt_fraction[I...]

        if (anomaly_ratio_ij > 0.5 || magma_ratio_ij > 0.5) && melt_fraction_ij ≥ threshold
            cells[I...] = true
        else
            cells[I...] = false
        end

        return nothing
    end

    ni = size(phase_ratios.center)
    return @parallel (@idx ni) _compute_cells_for_Q!(cells, threshold, phase_ratios.center, magma_phase, anomaly_phase, melt_fraction)

end

numcells(A::AbstractArray) = count(x -> x == 1.0, A)

function compute_thermal_source!(H, T_erupt, threshold, V_erupt, cells, ϕ, phase_ratios, dt, args, di, magma_phase, anomaly_phase, rheology)
    @parallel_indices (I...) function _compute_thermal_source!(H, T_erupt, threshold, V_erupt, cells, center_ratio, dt, args, dx, dy, magma_phase, anomaly_phase, heology)
        magma_ratio_ij = @index center_ratio[magma_phase, I...]
        anomaly_ratio_ij = @index center_ratio[anomaly_phase, I...]
        phase_ij = center_ratio[I...]
        total_fraction = magma_ratio_ij + anomaly_ratio_ij
        V_eruptij = V_erupt * ((total_fraction * cells[I...]) * inv(numcells(cells)))

        ϕ_ij = ϕ[I...]
        args_ij = (; T = args.T[I...], P = args.P[I...])
        Tij = args_ij.T
        ρCp = JustRelax.JustRelax2D.compute_ρCp(rheology, phase_ij, args_ij)

        if ((anomaly_ratio_ij > 0.5 || magma_ratio_ij > 0.5) && ϕ_ij ≥ threshold)
            H[I...] = ((V_eruptij / dt) * ρCp[I...] * (max(T_erupt - Tij, 0.0))) / (dx * dy * dx)
        #   [W/m^3] = [[m3/[s]] * [[kg/m^3] * [J/kg/K]] * [K]] / [[m] * [m] * [m]]
        end
        return nothing
    end

    ni = size(phase_ratios.center)

    @parallel (@idx ni) _compute_thermal_source!(H, T_erupt, threshold, V_erupt, cells, phase_ratios.center, dt, args, di..., magma_phase, anomaly_phase, rheology)
end



function compute_VEI!(V_erupt)
    if V_erupt <= 1.0e4
        return 0
    elseif V_erupt <= 1.0e6
        return 1
    elseif V_erupt <= 1.0e7
        return 2
    elseif V_erupt <= 1.0e8
        return 3
    elseif V_erupt <= 1.0e9
        return 4
    elseif V_erupt <= 1.0e10
        return 5
    elseif V_erupt <= 1.0e11
        return 6
    elseif V_erupt <= 1.0e12
        return 7
    else
        return 8
    end
end

function d18O_anomaly!(
    d18O, z, phase_ratios,
    magma_phase,
    anomaly_phase,
    lower_crust,
    air_phase;
    crust_gradient::Bool = true,
    crust_min::Float64 = -10.0,
    crust_max::Float64 = 3.0,
    crust_const::Float64 = 0.0,
    magma_const::Float64 = 5.5,

)
    ni = size(phase_ratios.vertex)

    @parallel_indices (i, j) function _d18O_anomaly!(d18O, z, vertex_ratio, magma_phase, anomaly_phase, lower_crust, air_phase)

        magma_ratio_ij = @index vertex_ratio[magma_phase, i, j]
        anomaly_ratio_ij = @index vertex_ratio[anomaly_phase, i, j]
        lower_crust_ratio_ij = @index vertex_ratio[lower_crust, i, j]
        air_ratio_ij = @index vertex_ratio[air_phase, i, j]

        if magma_ratio_ij > 0.5 || lower_crust_ratio_ij > 0.5 || anomaly_ratio_ij > 0.5
            d18O[i,j] = magma_const
        elseif air_ratio_ij > 0.5
            d18O[i,j] = 3.0
        elseif z[j] .> -3e3
            if crust_gradient
                # Linear gradient from crust_min at shallowest to crust_max at deepest
                zmin = z[1]
                zmax = z[end]
                d18O[i, j] = crust_min + (crust_max - crust_min) * (z[j] - zmin) / (zmax - zmin)
            else
                d18O[i, j] = crust_const
            end
        elseif z[j] .< -3e3
            d18O[i,j] = 5.0
        end
        return nothing
    end

    @parallel (@idx ni) _d18O_anomaly!(d18O, z, phase_ratios.vertex, magma_phase, anomaly_phase, lower_crust, air_phase)

    return nothing
end

# ## END OF HELPER FUNCTION ------------------------------------------------------------
# ## Custom Colormap by T.Keller
# using MAT
# using Colors, ColorSchemes

# matfile = matopen("SmallScaleCaldera/ocean.mat")
# my_cmap_data = read(matfile, "ocean") # e.g., "colormap_variable" is the variable name in the .mat file
# close(matfile)
# ocean = ColorScheme([RGB(r...) for r in eachrow(my_cmap_data)])
# ocean_rev = ColorScheme([RGB(r...) for r in eachrow(reverse(my_cmap_data, dims=1))])
# --------------------------------------------------------

## END OF MAIN SCRIPT ----------------------------------------------------------------
const plotting = true
const progressiv_extension = false
do_vtk = true # set to true to generate VTK files for ParaView

# conduit, depth, radius, ar, extension, fric_angle = parse.(Float64, ARGS[1:end])

# figdir is defined as Systematics_depth_radius_ar_extension
# figdir   = "Systematics/Caldera2D_$(today())_$(depth)_$(radius)_$(ar)_$(extension)_$(fric_angle)"
figdir = "Systematics/Caldera2D_$(today())"
n = 512
nx, ny = n, n >> 1

li, origin, phases_GMG, T_GMG, _, V_total, V_eruptible, layers, air_phase = setup2D(
    nx + 1, ny + 1;
    sticky_air = 4.0e0,
    dimensions = (40.0e0, 20.0e0), # extent in x and y in km
    flat = false, # flat or volcano cone
    chimney = false, # conduit or not
    layers = 3, # number of layers
    volcano_size = (3.0e0, 7.0e0),    # height, radius
    conduit_radius = 1.0e-2, # radius of the conduit
    chamber_T = 1050.0e0, # temperature of the chamber
    # chamber_depth  = depth, # depth of the chamber
    # chamber_radius = radius, # radius of the chamber
    # aspect_x       = ar, # aspect ratio of the chamber
)

igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end


# ## BEGIN OF MAIN SCRIPT --------------------------------------------------------------

# Physical domain ------------------------------------
ni = nx, ny           # number of cells
di = @. li / ni       # grid steps
grid = Geometry(ni, li; origin = origin)
(; xci, xvi) = grid             # nodes at the center and vertices of the cells
# ----------------------------------------------------

# Initialize particles -------------------------------
nxcell = 100
max_xcell = 150
min_xcell = 75
particles = init_particles(
    backend_JP, nxcell, max_xcell, min_xcell, xvi, di, ni
)

# Initialize marker chain
nxcell, max_xcell, min_xcell = 100, 150, 75
initial_elevation = 0.0e0
chain = init_markerchain(backend_JP, nxcell, min_xcell, max_xcell, xvi[1], initial_elevation)
# air_phase                    = 6
topo_y = extract_topo_from_GMG_phases(phases_GMG, xvi, air_phase)
fill_chain_from_vertices!(chain, PTArray(backend)(topo_y))

# rock ratios for variational stokes
# RockRatios
ϕ = RockRatio(backend, ni)
# update_rock_ratio!(ϕ, phase_ratios, air_phase)
compute_rock_fraction!(ϕ, chain, xvi, di)


# topo_y = chain.h_vertices
# nx, ny = size(ϕ.center)
# xv,yv = xvi;
# j = 255
# # cell origin
# ox = xv[i]
# oy = yv[j]

# p1 = GridGeometryUtils.Point(ox, topo_y[i])
# p2 = GridGeometryUtils.Point(xv[i + 1], topo_y[i + 1])

# s = Segment(p1, p2)

# r = Rectangle((ox, oy), di...)

# JustPIC._2D.cell_rock_area(s, r)

# @inline function is_chain_above_cell(s::Segment, r::Rectangle)
#     max_y = r.origin[2] + r.h
#     # Check if the segment is above the rectangle
#     return GridGeometryUtils.geq_r(s.p1[2], max_y) && GridGeometryUtils.geq_r(s.p2[2], max_y)
# end

# @inline function is_chain_below_cell(s::Segment, r::Rectangle)
#     min_y = r.origin[2]
#     # Check if the segment is below the rectangle
#     return GridGeometryUtils.leq_r(s.p1[2], min_y) && GridGeometryUtils.leq_r(s.p2[2], min_y)
# end

# function cell_rock_area(s::Segment, r::Rectangle{T}) where {T}
#     A = if is_chain_above_cell(s, r)
#         one(T)
#     elseif is_chain_below_cell(s, r)
#         zero(T)
#     else
#         # 3e0
#         intersecting_area2(s, r) 
#         # clamp(intersecting_area2(s, r) / area(r), zero(T), one(T))
#     end

#     return A
# end

# using GridGeometryUtils
# using StaticArrays

# function foo(ratio_center, chain, xvi, dxi)
#     topo_y = chain.h_vertices
#     nx, ny = size(ratio_center)
#     @parallel (1:nx, 1:ny) _compute_area_below_chain_center!(
#         ratio_center, topo_y, xvi..., dxi
#     )
#     return nothing
# end

# @parallel_indices (i, j) function _compute_area_below_chain_center!(
#         ratio::AbstractArray, topo_y, xv, yv, dxi
#     )

#     # cell origin
#     ox = xv[i]
#     oy = yv[j]

#     p1 = GridGeometryUtils.Point(ox, topo_y[i])
#     p2 = GridGeometryUtils.Point(xv[i + 1], topo_y[i + 1])
#     s = Segment(p1, p2)

#     r = Rectangle((ox, oy), dxi...)
#     # ratio[i, j] = 
#     cell_rock_area(s, r)

#     return nothing
# end

# foo(ϕ.center, chain, xvi, di)
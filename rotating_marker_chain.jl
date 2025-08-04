using CUDA
using JustPIC
using JustPIC._2D
using ParallelStencil

# @init_parallel_stencil(Threads, Float64, 2)
# const backend = JustPIC.CPUBackend

@init_parallel_stencil(CUDA, Float64, 2)
const backend = CUDABackend

# using GLMakie

function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = x1 - dx
    xF = x2 + dx
    return LinRange(xI, xF, n + 2)
end

# Analytical flow solution
vi_stream(x) = π * 1.0e-5 * (x - 0.5)

# function main()

# Initialize particles -------------------------------
# nxcell, max_xcell, min_xcell = 9, 9, 1
n = 51
nx = n - 1
Lx = Ly = 1.0
# nodal vertices
xvi = xv, yv = LinRange(0, Lx, n), LinRange(0, Ly, n)
dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
# nodal centers
xc, yc = LinRange(0 + dx / 2, Lx - dx / 2, n - 1), LinRange(0 + dy / 2, Ly - dy / 2, n - 1)
# staggered grid velocity nodal locations
grid_vx = xv, expand_range(yc)
grid_vy = expand_range(xc), yv
grid_vxi = grid_vx, grid_vy

# Cell fields -------------------------------
Vx = TA(backend)([-vi_stream(y) for x in grid_vx[1], y in grid_vx[2]])
Vy = TA(backend)([ vi_stream(x) for x in grid_vy[1], y in grid_vy[2]])

xc0 = yc0 = 0.25
R = 24 * dx
T = TA(backend)([ ((x - xc0)^2 + (y - yc0)^2 ≤ R^2) * 1.0 for x in xv, y in yv])
V = Vx, Vy

w = π * 1.0e-5  # angular velocity
period = 1  # revolution number
tmax = period / (w / (2 * π))
dt = 200.0

nxcell, min_xcell, max_xcell = 12, 6, 24
initial_elevation = Ly / 2
chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, initial_elevation)
method = RungeKutta2()
ratios = (;
    center = @zeros(nx, nx),
    vertex = @zeros(nx+1, nx+1),
    Vx     = @zeros(nx+1, nx),
    Vy     = @zeros(nx  , nx+1),
)

for i in 1:5
    @show i
    # compute_rock_fraction!(ratios, chain, xvi, dxi)
    advect_markerchain!(chain, method, V, grid_vxi, dt)
end

# compute_rock_fraction!(ratios, chain, xvi, dxi)


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
#     ratio[i, j] = JP.cell_rock_area(s, r)

#     return nothing
# end

using GridGeometryUtils
center = CUDA.zeros(nx, nx)
# center = zeros(nx, nx)

topo_y = chain.h_vertices
nx, ny = size(center)
import JustPIC._2D as JP

@parallel_indices (i, j) function foo(
        ratio::AbstractArray, topo_y, xv, yv, dxi
    )

    # cell origin
    ox = xv[i]
    oy = yv[j]

    # p1 = GridGeometryUtils.Point(ox, oy)
    p1 = GridGeometryUtils.Point(ox, topo_y[i])
    p2 = GridGeometryUtils.Point(xv[i + 1], topo_y[i + 1])
    s = Segment(p1, p2)

    r = Rectangle((ox, oy), dxi...)
    # ratio[i, j] = rand() 
    ratio[i, j] = JP.cell_rock_area(s, r)
    # # CUDA.@cushow JP.cell_rock_area(s, r)
    # @show JP.cell_rock_area(s, r)

    return nothing
end

@parallel (1:nx, 1:ny) foo(
    center, topo_y, xvi..., dxi
)

# f = Figure()
# ax = Axis(f[1, 1])
# # # vector of shapes
# # poly!(
# #     ax,
# #     Rect(0, 0, 1, 1),
# #     color = :lightgray,
# # )
# px = Array(chain.coords[1].data[:])
# py = Array(chain.coords[2].data[:])
# scatter!(ax, px, py, color = :black)
# display(f)

##

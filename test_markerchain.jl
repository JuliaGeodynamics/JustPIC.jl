const isGPU = false
@static if isGPU 
    using CUDA
end
using JustPIC, JustPIC._2D

using ParallelStencil

@static if isGPU 
    const backend = CUDABackend
    @init_parallel_stencil(Threads, Float64, 2)

else
    const backend = JustPIC.CPUBackend
    @init_parallel_stencil(CUDA, Float64, 2)

end

# using GLMakie

# model geometry
n        = 51
nx       = n-1
Lx       = Ly = 1.0
dxi      = dx, dy = Lx/nx, Ly/nx
# nodal vertices
xvi      = xv, yv = LinRange(0, Lx, n), LinRange(0, Ly, n)

# initialize the marker chain
nxcell, min_xcell, max_xcell = 12, 6, 24
initial_elevation = Ly/2
chain             = init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, initial_elevation);

init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, initial_elevation);

# create topographic profile
nt = 2000
topo_x = TA(backend)(LinRange(0, 1, nt))
topo_y = TA(backend)(sin.(2π*topo_x) .* 0.1)

# fill the chain with the topographic profile
fill_chain_from_chain!(chain, topo_x, topo_y)

# plot the chain
compute_topography_vertex!(chain)

# create topographic profile @ vertices
topo_x = TA(backend)(LinRange(0, 1, n))
topo_y = TA(backend)(sin.(2π*topo_x) .* 0.1)
chain.h_vertices .= topo_y

fill_chain_from_vertices!(chain, topo_y)

x_chain = Array(chain.coords[1].data[:])
y_chain = Array(chain.coords[2].data[:])
perms = sortperm(x_chain)

scatterlines(Array(chain.cell_vertices), Array(chain.h_vertices), color=:red)
scatter!(x_chain, y_chain, color=:black)

####
chain.h_vertices .= 0

compute_topography_vertex!(chain)
scatterlines(chain.cell_vertices, chain.h_vertices, color=:red)

# initialize the marker chain
# create topographic profile
topo_x = TA(backend)(LinRange(0, 1, n))
topo_y = TA(backend)(sin.(2π*topo_x) .* 0.1)

@edit fill_chain_from_vertices!(chain, topo_y)

fill_chain_from_vertices!(chain, topo_y)

###############

compute_area_below_chain_centers!(ratio_center, chain, xvi, dxi)

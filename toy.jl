using JustPIC
using JustPIC._2D
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

backend = CPUBackend
L  = 1
n  = 32 # num cells
xv = LinRange(0, L, n+1)
dx = L/n
nxcell = 8
min_xcell, max_xcell = 6, 10
initial_elevation = 0.0

chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, initial_elevation, dx);

move_particles!(chain)

function foo(backend, nxcell, min_xcell, max_xcell, x, initial_elevation, dx)
    
    @parallel_indices (i) function fill_coords_index!(
        px, py, index, x, initial_elevation, dx_chain, nxcell, max_xcell
    )
        # lower-left corner of the cell
        x0 = x[i]
        # fill index array
        for ip in 1:nxcell
            @cell px[ip, i] = x0 + dx_chain * ip
            @cell py[ip, i] = initial_elevation
            @cell index[ip, i] = true
        end
        return nothing
    end

    nx = length(x) - 1
    dx_chain = dx / (nxcell + 1)
    px, py = ntuple(_ -> @fill(NaN, (nx,), celldims = (max_xcell,)), Val(2))
    index = @fill(false, (nx,), celldims = (max_xcell,), eltype = Bool)

    @parallel (1:nx) fill_coords_index!(
        px, py, index, x, initial_elevation, dx_chain, nxcell, max_xcell
    )

    return MarkerChain(backend, (px, py), index, x, min_xcell, max_xcell)
end

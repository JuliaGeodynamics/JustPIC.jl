using JustPIC, JustPIC._2D

# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA"),
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU")
const backend = JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

# using GLMakie
using TimerOutputs

function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = x1 - dx
    xF = x2 + dx
    return LinRange(xI, xF, n + 2)
end

# Analytical flow solution
vx_stream(x, y) =  250 * sin(π*x) * cos(π*y)
vy_stream(x, y) = -250 * cos(π*x) * sin(π*y)
g(x) = Point2f(
    vx_stream(x[1], x[2]),
    vy_stream(x[1], x[2])
)

function main()
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 24, 48, 28
    n = 256
    nx = ny = n-1
    Lx = Ly = 1.0
    # nodal vertices
    xvi = xv, yv = range(0, Lx, length=n), range(0, Ly, length=n)
    dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
    # nodal centers
    xc, yc = range(0+dx/2, Lx-dx/2, length=n-1), range(0+dy/2, Ly-dy/2, length=n-1)
    # staggered grid velocity nodal locations
    grid_vx = xv, expand_range(yc)
    grid_vy = expand_range(xc), yv

    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi...)

    # Cell fields -------------------------------
    Vx = TA(backend)([vx_stream(x, y) for x in grid_vx[1], y in grid_vx[2]]);
    Vy = TA(backend)([vy_stream(x, y) for x in grid_vy[1], y in grid_vy[2]]);
    T  = TA(backend)([y for x in xv, y in yv]);
    V  = Vx, Vy;

    dt = min(dx / maximum(abs.(Array(Vx))),  dy / maximum(abs.(Array(Vy))));
    dt *= 0.25

    # Advection test
    particle_args = pT, = init_cell_arrays(particles, Val(1));
    grid2particle!(pT, xvi, T, particles);

    !isdir("figs") && mkdir("figs")

    niter = 5
    for it in 1:niter
        to = TimerOutput()
        @timeit to "advect" advection!(particles, RungeKutta2(2/3), V, (grid_vx, grid_vy), dt)
        @timeit to "move" move_particles!(particles, xvi, particle_args)
        @timeit to "injection" inject_particles!(particles, (pT, ), xvi)
        @timeit to "p2g" particle2grid!(T, pT, xvi, particles)
        @timeit to "g2p" grid2particle!(pT, xvi, T, particles);
        @show to

        if rem(it, 10) == 0
            f, ax, = heatmap(xvi..., Array(T), colormap=:batlow)
            streamplot!(ax, g, xvi...)
            save("figs/test_$(it).png", f)
            f
        end
    end

    println("Finished")
end

main()

import JustPIC._2D.corner_field_nodes

p_i = particles.coords[1][10,10][1], particles.coords[2][10,10][1]
idx = 10,10
xi_vx = grid_vx

@b corner_field_nodes($(T, p_i, xi_vx, dxi, idx)...)
corner_field_nodes(T, p_i, xi_vx, dxi, idx)

@generated function foo(
    particle,
    xi_vx,
    dxi,
    idx::Union{SVector{N,Integer},NTuple{N,Integer}},
) where {N}
    quote
        Base.@_inline_meta
        Base.@nexprs $N i -> begin
            # unpack
            corrected_idx_i = idx[i]
            # compute offsets and corrections
            corrected_idx_i += @inline JustPIC._2D.vertex_offset(
                xi_vx[i][corrected_idx_i], particle[i], dxi[1]
            )
            cell_i = xi_vx[i][corrected_idx_i]
        end

        cells = Base.@ncall $N tuple cell

        return cells
    end
end
@b foo($(p_i, xi_vx, dxi, idx)...)

foo(p_i, xi_vx, dxi, idx)
@code_warntype foo(p_i, xi_vx, dxi, idx)

@b cell_index($(p_i[1], dxi[1])...)

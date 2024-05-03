using JustPIC
using JustPIC._2D

# Threads is the default backend, 
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA"), 
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU")
const backend = JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

using GLMakie
using ImplicitGlobalGrid

# Analytical flow solution
vx_stream(x, y) =  250 * sin(π*x) * cos(π*y)
vy_stream(x, y) = -250 * cos(π*x) * sin(π*y)
g(x) = Point2f(
    vx_stream(x[1], x[2]),
    vy_stream(x[1], x[2])
)

function main()
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 8, 8, 8
    n   = 32
    nx  = ny = n-1
    me, dims, = init_global_grid(n-1, n-1, 0; init_MPI=JustPIC.MPI.Initialized() ? false : true)
    Lx  = Ly = 1.0
    dxi = dx, dy = Lx /(nx_g()-1), Ly / (ny_g()-1)
    # nodal vertices
    xvi = xv, yv = let
        dummy = zeros(n, n) 
        xv  = TA([x_g(i, dx, dummy) for i in axes(dummy, 1)])
        yv  = TA([y_g(i, dx, dummy) for i in axes(dummy, 2)])
        xv, yv
    end
    # nodal centers
    xci = xc, yc = let
        dummy = zeros(nx, ny) 
        xc  = @zeros(nx) 
        xc .= TA([x_g(i, dx, dummy) for i in axes(dummy, 1)])
        yc  = TA([y_g(i, dx, dummy) for i in axes(dummy, 2)])
        xc, yc
    end

    # staggered grid for the velocity components
    grid_vx = xv, add_ghost_nodes(yc, dy, (0.0, Ly))
    grid_vy = add_ghost_nodes(xc, dx, (0.0, Lx)), yv
    
    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi..., dxi..., nx, ny
    )

    # Cell fields -------------------------------
    Vx = TA(backend)([vx_stream(x, y) for x in grid_vx[1], y in grid_vx[2]])
    Vy = TA(backend)([vy_stream(x, y) for x in grid_vy[1], y in grid_vy[2]])
    T  = TA(backend)([y for x in xv, y in yv])
    T0 = deepcopy(T) 
    V  = Vx, Vy

    nx_v = (size(T, 1)-2)*dims[1];
    ny_v = (size(T, 2)-2)*dims[2];
    T_v  = zeros(nx_v, ny_v)
    T_nohalo = @zeros(size(T).-2)

    dt = min(dx / maximum(abs.(Vx)),  dy / maximum(abs.(Vy)))

    # Advection test
    particle_args = pT, = init_cell_arrays(particles, Val(1))
    grid2particle!(pT, xvi, T, particles)
    
    niter = 250
    for iter in 1:niter
        me == 0 && @show iter
        
        # grid2particle!(pT, xvi, T, T0, particles)

        # advect particles
        advection!(particles, RungeKutta2(2/3), V, (grid_vx, grid_vy), dt)

        # update halos
        update_cell_halo!(particles.coords..., particle_args...);
        update_cell_halo!(particles.index)
        # shuffle particles
        move_particles!(particles, xvi, particle_args)
        # interpolate T from particle to grid
        particle2grid!(T, pT, xvi, particles)
        # T0 .= deepcopy(T) 

        @views T_nohalo .= T[2:end-1, 2:end-1]
        gather!(T_nohalo, T_v)

        if me == 0 && iter % 10 == 0
            x_global = range(0, Lx, length=size(T_v,1))
            y_global = range(0, Ly, length=size(T_v,2))
            f, ax, = heatmap(x_global, y_global, T_v)
            w = 0.504
            offset = 0.5-(w-0.5)
            lines!(ax,
                [0, w, w, 0, 0],
                [0, 0, w, w, 0],
                linewidth = 3
            )
            lines!(ax,
                [0, w, w, 0, 0] .+ offset,
                [0, 0, w, w, 0],
                linewidth = 3
            )
            lines!(ax,
                [0, w, w, 0, 0] .+ offset,
                [0, 0, w, w, 0] .+ offset,
                linewidth = 3
            )
            lines!(ax,
                [0, w, w, 0, 0],
                [0, 0, w, w, 0] .+ offset,
                linewidth = 3
            )
   
            save("figs/T_MPI_$iter.png", f)
            
        end    

        # px = particles.coords[1].data[:]
        # py = particles.coords[1].data[:]
        # idx = particles.index.data[:]
        # f, ax, = lines(
        #         [0, w, w, 0, 0],
        #         [0, 0, w, w, 0],
        #         linewidth = 3
        #     )
        #     lines!(ax,
        #         [0, w, w, 0, 0] .+ offset,
        #         [0, 0, w, w, 0],
        #         linewidth = 3
        #     )
        #     lines!(ax,
        #         [0, w, w, 0, 0] .+ offset,
        #         [0, 0, w, w, 0] .+ offset,
        #         linewidth = 3
        #     )
        #     lines!(ax,
        #         [0, w, w, 0, 0],
        #         [0, 0, w, w, 0] .+ offset,
        #         linewidth = 3
        #     )
        # scatter(px[idx], py[idx], color=:black)
        # save("figs$me/particles_$iter.png", f)

    end
    
    # f, ax, = heatmap(xvi..., T, colormap=:batlow)
    # streamplot!(ax, g, xvi...)
    # f
end

main()

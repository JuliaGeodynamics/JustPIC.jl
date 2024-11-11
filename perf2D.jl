const isGPU = false

using JustPIC
using JustPIC._2D

# Threads is the default backend, 
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA"), 
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU")
const backend = @static if isGPU
     CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

using CSV, DataFrames

function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = round(x1-dx; sigdigits=5)
    xF = round(x2+dx; sigdigits=5)
    range(xI, xF, length=n+2)
end

# Analytical flow solution
vx_stream(x, y) =  250 * sin(π*x) * cos(π*y)
vy_stream(x, y) = -250 * cos(π*x) * sin(π*y)
g(x) = Point2f(
    vx_stream(x[1], x[2]),
    vy_stream(x[1], x[2])
)

function main(; n = 256, fn_advection = advection!)
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 25, 50, 5
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
        backend, nxcell, max_xcell, min_xcell, xvi...,
    )

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
    
    !isdir("perf") && mkdir("perf")

    adv    = Float64[]
    move   = Float64[]
    inject = Float64[]
    p2g    = Float64[]
    g2p    = Float64[]
    ttot   = Float64[]
    np     = Int64[]

    niter  = 250

    for _ in 1:niter
        t_adv    = @elapsed advection!(particles, RungeKutta2(), V, (grid_vx, grid_vy), dt)
        t_move   = @elapsed move_particles!(particles, xvi, particle_args)
        t_inject = @elapsed inject_particles!(particles, (pT, ), xvi)
        t_p2g    = @elapsed particle2grid!(T, pT, xvi, particles)
        t_g2p    = @elapsed grid2particle!(pT, xvi, T, particles)
        t_ttot   = t_adv + t_move + t_inject + t_p2g + t_g2p
        
        push!(adv,    t_adv)
        push!(move,   t_move)
        push!(inject, t_inject)
        push!(p2g,    t_p2g)
        push!(g2p,    t_g2p)
        push!(ttot,   t_ttot)

        push!(np, count(particles.index.data))
    end

    df = DataFrame(
        ttotal    = ttot,
        advection = adv,
        move      = move,
        inject    = inject,
        p2g       = p2g,
        g2p       = g2p,
        np        = np
    )

    adv_interp = if fn_advection === advection!
        "linear"
    elseif fn_advection === advection_LinP!
        "LinP"
    else fn_advection === advection_MQS!
        "MQS"
    end

    if isGPU
        CSV.write("perf/perf2D_$(adv_interp)_n$(n)_CUDA.csv", df)
    else
        CSV.write("perf/perf2D_$(adv_interp)_n$(n)_nt_$(Threads.nthreads()).csv", df)
    end
    println("Finished: n = $n")
end

function runner()
    n = 64, 128, 256, 512, 1024

    fn_advection = advection!, advection_LinP!, advection_MQS! 
    for n in n
        Base.@nexprs 3 i-> main(n=n, fn_advection=fn_advection[i])
    end
end

runner()
using JustPIC
using JustPIC._2D
using GLMakie

# Threads is the default backend, 
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA"), 
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU")
const backend = JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = round(x1-dx; sigdigits=5)
    xF = round(x2+dx; sigdigits=5)
    range(xI, xF, length=n+2)
end

# Analytical flow solution
vx_stream(x, y, r) = ((x-0.5)^2+(y-0.5)^2 > r^2 ?  250 * sin(π*x) * cos(π*y) : -250 * sin(π*x) * cos(π*y))
vy_stream(x, y, r) = ((x-0.5)^2+(y-0.5)^2 > r^2 ? -250 * cos(π*x) * sin(π*y) : 250 * cos(π*x) * sin(π*y)) 
# vx_stream(x, y, r) = ((x-0.5)^2+(y-0.5)^2 > r^2 ?  250 * sin(π*x) * cos(π*y) : 0e0)
# vy_stream(x, y, r) = ((x-0.5)^2+(y-0.5)^2 > r^2 ? -250 * cos(π*x) * sin(π*y) : 0e0)

# vx_stream(x, y, r) =  250 * sin(π*x) * cos(π*y)
# vy_stream(x, y, r) = -250 * cos(π*x) * sin(π*y)
g(x) = Point2f(
    vx_stream(x[1], x[2], 0.25),
    vy_stream(x[1], x[2], 0.25)
)

function main()
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 25, 75, 15
    n = 41
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
    particles_MQS = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi...,
    )

    # Cell fields -------------------------------
    Vx = TA(backend)([vx_stream(x, y, 0.25) for x in grid_vx[1], y in grid_vx[2]]);
    Vy = TA(backend)([vy_stream(x, y, 0.25) for x in grid_vy[1], y in grid_vy[2]]);
    T  = TA(backend)([y for x in xv, y in yv]);
    V  = Vx, Vy;

    np = particles.np # number of passive markers
    passive_coords = ntuple(Val(2)) do i
        (rand(np) .+ 1) .* Lx/4
    end

    passive_markers = init_passive_markers(backend, passive_coords);
    T_marker = TA(backend)(zeros(np))
    P_marker = TA(backend)(zeros(np))


    dt  = min(dx / maximum(abs.(Array(Vx))),  dy / maximum(abs.(Array(Vy))));
    dt *= 0.1 
    # Advection test
    pT, pT_MQS, pT_passive = init_cell_arrays(particles, Val(3));
    grid2particle!(pT, xvi, T, particles);
    grid2particle!(pT_MQS, xvi, T, particles);
    grid2particle!(pT_passive, xvi, T, particles);
    
    adv_MQS = Float64[]
    move_MQS = Float64[]
    g2p_MQS = Float64[]
    p2g_MQS = Float64[]
    adv = Float64[]
    move = Float64[]
    g2p = Float64[]
    p2g = Float64[]

    adv_pass = Float64[]
    g2p_pass = Float64[]
    p2g_pass = Float64[]
    
    buffer = similar(T)

    niter = 1000
    @show sum(particles.index.data)

    for it in 1:niter
        push!(adv_MQS, @elapsed advection!(particles_MQS, RungeKutta2(), V, (grid_vx, grid_vy), dt))
        # advection_LinP!(particles, RungeKutta2(), V, (grid_vx, grid_vy), dt)
        # push!(adv_MQS, @elapsed advection_MQS!(particles_MQS, RungeKutta2(), V, (grid_vx, grid_vy), dt))
        push!(move_MQS, @elapsed move_particles!(particles_MQS, xvi, (pT_MQS,)))
        push!(g2p_MQS, @elapsed grid2particle!(pT_MQS, xvi, T, particles);)
        push!(p2g_MQS, @elapsed particle2grid!(T, pT_MQS, xvi, particles);)

        push!(adv, @elapsed advection_MQS!(particles, RungeKutta2(), V, (grid_vx, grid_vy), dt))
        push!(move, @elapsed move_particles!(particles, xvi, (pT,)))
        push!(g2p, @elapsed grid2particle!(pT, xvi, T, particles);)
        push!(p2g, @elapsed particle2grid!(T, pT, xvi, particles);)

        push!(adv_pass, @elapsed advection!(passive_markers, RungeKutta2(), V, (grid_vx, grid_vy), dt))
        push!(g2p_pass, @elapsed grid2particle!(T_marker, xvi, T, passive_markers))
        push!(p2g_pass, @elapsed particle2grid!(T, T_marker, buffer, xvi, passive_markers))
    end
    t_MQS    = (; adv = adv_MQS, move = move_MQS, g2p = g2p_MQS, p2g = p2g_MQS)
    t_normal = (; adv = adv,     move = move,     g2p = g2p, p2g = p2g)
    t_pass   = (; adv = adv_pass,                 g2p = g2p_pass, p2g = p2g_pass)
    @show sum(particles.index.data)
    return t_MQS, t_normal, t_pass
end

t_MQS, t_normal, t_pass = main();
 
f,ax,l=scatter(t_pass.adv[2:end] ./ t_normal.adv[2:end], label = "opt")
# f,ax,l=scatter(t_normal.adv[2:end], label = "opt")
# scatter!(ax,t_pass.adv[2:end], label = "classic")
# axislegend(ax)
# f
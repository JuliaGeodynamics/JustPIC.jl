
# const isGPU = false
@show ARGS[1]
const isGPU = ARGS[1] === "true" ? true : false

@static if isGPU
    using CUDA
end

using JustPIC
using JustPIC._3D

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
vz_stream(x, z) = -250 * cos(π*x) * sin(π*z)

function main(; n = 256, fn_advection = advection!)
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 125, 250, 5
    nx  = ny = nz = n-1
    Lx  = Ly = Lz = 1.0
    ni  = nx, ny, nz
    Li  = Lx, Ly, Lz
    # nodal vertices
    xvi = xv, yv, zv = ntuple(i -> range(0, Li[i], length=n), Val(3))
    # grid spacing
    dxi = dx, dy, dz = ntuple(i -> xvi[i][2] - xvi[i][1], Val(3))
    # nodal centers
    xci = xc, yc, zc = ntuple(i -> range(0+dxi[i]/2, Li[i]-dxi[i]/2, length=ni[i]), Val(3))

    # staggered grid velocity nodal locations
    grid_vx = xv              , expand_range(yc), expand_range(zc)
    grid_vy = expand_range(xc), yv              , expand_range(zc)
    grid_vz = expand_range(xc), expand_range(yc), zv

    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi...,
    )

    # Cell fields -------------------------------
    Vx = TA(backend)([vx_stream(x, z) for x in grid_vx[1], y in grid_vx[2], z in grid_vx[3]])
    Vy = TA(backend)([vy_stream(x, z) for x in grid_vy[1], y in grid_vy[2], z in grid_vy[3]])
    Vz = TA(backend)([vz_stream(x, z) for x in grid_vz[1], y in grid_vz[2], z in grid_vz[3]])
    T  = TA(backend)([z for x in xv, y in yv, z in zv])
    V  = Vx, Vy, Vz

    dt = min(dx / maximum(abs.(Vx)), dy / maximum(abs.(Vy)), dz / maximum(abs.(Vz))) / 2

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

    niter  = 25

    for _ in 1:niter
        t_adv    = @elapsed advection!(particles, RungeKutta2(), V, (grid_vx, grid_vy, grid_vz), dt)
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

    @show isGPU
    if isGPU
        CSV.write("perf3D/perf3D_$(adv_interp)_n$(n)_CUDA.csv", df)
    else
        CSV.write("perf3D/perf3D_$(adv_interp)_n$(n)_nt_$(Threads.nthreads()).csv", df)
    end
    println("Finished: n = $n")
end

function runner()
    n = 64, 128 , 256, 512, 1024

    fn_advection = advection!, advection_LinP!, advection_MQS! 
    for n in n
        Base.@nexprs 3 i-> main(n=n, fn_advection=fn_advection[i])
    end
end

runner()
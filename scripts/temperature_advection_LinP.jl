using JustPIC
using JustPIC._2D

# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA"),
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU")
const backend = JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

using GLMakie

function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = x1 - dx
    xF = x2 + dx
    return LinRange(xI, xF, n + 2)
end
# Analytical flow solution
vx_stream(x, y) = 250 * sin(π * x) * cos(π * y)
vy_stream(x, y) = -250 * cos(π * x) * sin(π * y)
g(x) = Point2f(
    vx_stream(x[1], x[2]),
    vy_stream(x[1], x[2])
)

function main()
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 24, 48, 28
    n = 64 #256
    nx = ny = n - 1
    Lx = Ly = 1.0
    # nodal vertices
    xvi = xv, yv = range(0, Lx, length = n), range(0, Ly, length = n)
    dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
    # nodal centers
    xc, yc = range(0 + dx / 2, Lx - dx / 2, length = n - 1), range(0 + dy / 2, Ly - dy / 2, length = n - 1)
    # staggered grid velocity nodal locations
    grid_vx = xv, expand_range(yc)
    grid_vy = expand_range(xc), yv

    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi...,
    )

    # Cell fields -------------------------------
    Vx = TA(backend)([vx_stream(x, y) for x in grid_vx[1], y in grid_vx[2]])
    Vy = TA(backend)([vy_stream(x, y) for x in grid_vy[1], y in grid_vy[2]])
    T = TA(backend)([y for x in xv, y in yv])
    V = Vx, Vy

    dt = min(dx / maximum(abs.(Array(Vx))), dy / maximum(abs.(Array(Vy))))
    dt *= 0.6

    # Advection test
    particle_args = pT, = init_cell_arrays(particles, Val(1))
    grid2particle!(pT, xvi, T, particles)

    fname = "figs_LinP"
    # fname = "figs"
    !isdir(fname) && mkdir(fname)

    niter = 250
    for it in 1:niter
        # advection!(particles, RungeKutta2(2/3), V, (grid_vx, grid_vy), dt)
        advection_LinP!(particles, RungeKutta2(2 / 3), V, (grid_vx, grid_vy), dt)
        move_particles!(particles, xvi, particle_args)
        inject_particles!(particles, (pT,), xvi)
        particle2grid!(T, pT, xvi, particles)

        if rem(it, 10) == 0
            f, ax, = heatmap(xvi..., Array(T), colormap = :batlow)
            streamplot!(ax, g, xvi...)
            save(
                joinpath(fname, "test_$(it).png"),
                f
            )
            f
        end
    end

    return println("Finished")
end

main()

v = tuple(rand(6)...)
t = tuple(rand(2)...)

@inline lerp(v, t::NTuple{nD, T}) where {nD, T} = lerp(v, t, 0, Val(nD - 1))
@inline lerp(v, t::NTuple{nD, T}, i, ::Val{N}) where {nD, N, T} = lerp(t[N + 1], lerp(v, t, i, Val(N - 1)), lerp(v, t, i + 2^N, Val(N - 1)))
@inline lerp(v, t::NTuple{nD, T}, i, ::Val{0}) where {nD, T} = lerp(t[1], v[i + 1], v[i + 2])
@inline lerp(t::T, v0::T, v1::T) where {T <: Real} = fma(t, v1, fma(-t, v0, v0))


# @inline MQS(v, t::NTuple{nD,T})              where {nD,T}    = MQS(v, t, 0, Val(nD - 1))
# @inline MQS(v, t::NTuple{nD,T}, i, ::Val{N}) where {nD,N,T}  = lerp(t[N + 1], MQS(v, t, i, Val(N - 1)), MQS(v, t, i + 2^N, Val(N - 1)))
# @inline MQS(v, t::NTuple{nD,T}, i, ::Val{0}) where {nD,T}    = MQS(t[1], v[i + 1], v[i + 2], v[i + 3])

@inline function MQS(t::T, v0::T, v1::T, v2::T) where {T <: Real}
    linear_term = muladd(t, v2, muladd(-t, v1, v1))
    quadratic_correction = 0.5 * (t - 1.5)^2 * (muladd(-2, v1, v0) + v2)
    return linear_term + quadratic_correction
end

function MQS_Vx(v_cell, v_MQS, t::NTuple{2})
    # MQS for Vx - bottom
    v_bot = MQS(t[1], v_MQS[1], v_cell[1:2]...)
    # MQS for Vx - top
    v_top = MQS(t[1], v_MQS[2], v_cell[3:4]...)

    return vt = lerp(t[2], v_bot, v_top)
end

function MQS_Vy(v_cell, v_MQS, t::NTuple{2})
    # MQS for Vy - left
    v_left = MQS(t[2], v_MQS[1], v_cell[1], v_cell[3])
    # MQS for Vy - right
    v_right = MQS(t[2], v_MQS[2], v_cell[2], v_cell[4])

    return vt = lerp(t[1], v_left, v_right)
end
v = tuple(rand(6)...)
t = tuple(rand(2)...) 

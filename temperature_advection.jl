using JustPIC
using JustPIC._2D

using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)
# Threads is the default backend, 
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA"), 
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU")
const backend = JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

using GLMakie

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
# g(x) = Point2f(
#     vx_stream(x[1], x[2]),
#     vy_stream(x[1], x[2])
# )


function init_particles_noRNG(backend, nxcell, max_xcell, min_xcell, xvi::Vararg{N,T}) where {N,T}
    di = JustPIC._2D.compute_dx(xvi)
    ni = @. length(xvi) - 1

    return _init_particles(backend, nxcell, max_xcell, min_xcell, xvi, di, ni)
end

@inline function new_particle(xvi::NTuple{2}, di::NTuple{2}, ctr, np)
    th = (2 * pi) / np * (ctr - 1)
    r = min(di...) * 0.25
    p_new = (
        muladd(di[1], 0.5, muladd(r, cos(th), xvi[1])),
        muladd(di[2], 0.5, muladd(r, sin(th), xvi[2])),
    )
    return p_new
end

circle(r, c, θ) =(c[1] + r*sin(θ), c[2] + r*cos(θ))

function _init_particles(
    backend,
    nxcell,
    max_xcell,
    min_xcell,
    coords::NTuple{N,AbstractArray},
    dxᵢ::NTuple{N,T},
    nᵢ::NTuple{N,I},
) where {N,T,I}
    ncells = prod(nᵢ)
    np = max_xcell * ncells
    pxᵢ = ntuple(_ -> @rand(nᵢ..., celldims = (max_xcell,)), Val(N))
    index = @fill(false, nᵢ..., celldims = (max_xcell,), eltype = Bool)

    @parallel_indices (I...) function fill_coords_index(
        pxᵢ::NTuple{N,T}, index, coords, dxᵢ, nxcell, max_xcell
    ) where {N,T}
        # lower-left corner of the cell
        x0ᵢ = ntuple(Val(N)) do ndim
            coords[ndim][I[ndim]]
        end
        θ = LinRange(0, 2π, nxcell)
        r = min(dxᵢ...) * 0.75
        c = ntuple(Val(N)) do ndim
            x0ᵢ[ndim] + 0.5 * dxᵢ[ndim]
        end
        # fill index array
        for l in 1:max_xcell
            if l ≤ nxcell
                ntuple(Val(N)) do ndim
                    @cell pxᵢ[ndim][l, I...] = circle(r, c, θ[l])[ndim]
                end
                @cell index[l, I...] = true

            else
                ntuple(Val(N)) do ndim
                    @cell pxᵢ[ndim][l, I...] = NaN
                end
            end
        end
        return nothing
    end

    @parallel (JustPIC._2D.@idx nᵢ) fill_coords_index(pxᵢ, index, coords, dxᵢ, nxcell, max_xcell)

    return Particles(backend, pxᵢ, index, nxcell, max_xcell, min_xcell, np)
end


function main()
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 6, 20, 3
    n = 32
    # nx = ny = n-1
    nx = n-1
    ny = 2*n-1
    Lx = Ly = 1.0
    # nodal vertices
    xvi = xv, yv = range(0, Lx, length=n), range(0, Ly, length=n)
    dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
    # nodal centers
    xc, yc = range(0+dx/2, Lx-dx/2, length=n-1), range(0+dy/2, Ly-dy/2, length=n-1)
    # staggered grid velocity nodal locations
    grid_vx = xv, expand_range(yc)
    grid_vy = expand_range(xc), yv

    # particles = init_particles(
    #     backend, nxcell, max_xcell, min_xcell, xvi...,
    # )
    particles = init_particles_noRNG(
        backend, nxcell, max_xcell, min_xcell, xvi...,
    )
    
    # Cell fields -------------------------------
    Vx = TA(backend)([vx_stream(x, y) for x in grid_vx[1], y in grid_vx[2]]);
    Vy = TA(backend)([vy_stream(x, y) for x in grid_vy[1], y in grid_vy[2]]);
    T  = TA(backend)([y for x in xv, y in yv]);
    V  = Vx, Vy;

    dt = min(dx / maximum(abs.(Array(Vx))),  dy / maximum(abs.(Array(Vy))));
    # dt *= 0.25

    # Advection test
    particle_args = pT, = init_cell_arrays(particles, Val(1));
    grid2particle!(pT, xvi, T, particles);
    
    !isdir("figs") && mkdir("figs")

    niter = 250
    for it in 1:niter
        # advection!(particles, RungeKutta2(), V, (grid_vx, grid_vy), dt)
        advection_MQS!(particles, RungeKutta2(), V, (grid_vx, grid_vy), dt)
        # advection_LinP!(particles, RungeKutta2(), V, (grid_vx, grid_vy), dt)
        move_particles!(particles, xvi, particle_args)
        # inject_particles!(particles, (pT, ), xvi)

        particle2grid!(T, pT, xvi, particles)

    end
    pxx, pyy = particles.coords
    # f, = heatmap(xvi..., Array(T), colormap=:batlow)
    f, = scatter(pxx.data[:], pyy.data[:], markersize=4)
    # save("figs/test_$(it).png", f)

    println("Finished")
    f
end

main()

